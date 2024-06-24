package internal

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	"github.com/uptrace/bun"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/determined-ai/determined/master/internal/api"
	"github.com/determined-ai/determined/master/internal/authz"
	"github.com/determined-ai/determined/master/internal/command"
	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/internal/grpcutil"
	"github.com/determined-ai/determined/master/internal/rm/kubernetesrm"
	"github.com/determined-ai/determined/master/internal/templates"
	"github.com/determined-ai/determined/master/internal/workspace"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/schemas"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
	"github.com/determined-ai/determined/master/pkg/set"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/projectv1"
	"github.com/determined-ai/determined/proto/pkg/workspacev1"
)

const (
	// defaultKubernetesNamespace is the default Kubernetes namespace for a given cluster.
	defaultKubernetesNamespace = "default"
	staleLabel                 = "(stale)"
)

func maskStorageConfigSecrets(w *workspacev1.Workspace) error {
	if w.CheckpointStorageConfig == nil {
		return nil
	}

	// Convert to expconf.
	bytes, err := w.CheckpointStorageConfig.MarshalJSON()
	if err != nil {
		return err
	}
	var checkpointStorageConfig expconf.CheckpointStorageConfig
	if err = (&checkpointStorageConfig).UnmarshalJSON(bytes); err != nil {
		return err
	}

	// Convert back to proto.Struct with .Printable() called.
	bytes, err = checkpointStorageConfig.Printable().MarshalJSON()
	if err != nil {
		return err
	}
	if err = w.CheckpointStorageConfig.UnmarshalJSON(bytes); err != nil {
		return err
	}
	return nil
}

func (a *apiServer) validateRequestClusterName(clusterName string) (string, error) {
	if len(clusterName) == 0 && len(a.m.allRms) > 1 {
		return "", status.Errorf(codes.InvalidArgument, "must specify a cluster name when using MultiRM")
	}

	// Since SingleRM clusters don't have to specify the cluster name in namespace-binding requests,
	// we populate the cluster name appropriately if it is set in the master config.
	if len(a.m.allRms) == 1 && clusterName == "" {
		for k := range a.m.allRms {
			clusterName = k
		}
	}
	_, ok := a.m.allRms[clusterName]
	if !ok {
		return "", status.Errorf(codes.InvalidArgument,
			"no resource manager with cluster name %s", clusterName)
	}
	return clusterName, nil
}

func (a *apiServer) validateClusterNamespaceMeta(
	namespaceMeta map[string]*workspacev1.WorkspaceNamespaceMeta) (
	map[string]*workspacev1.WorkspaceNamespaceMeta, error,
) {
	allClusters := make(map[string]int)
	for clusterName, metadata := range namespaceMeta {
		namespace := metadata.Namespace
		if _, ok := allClusters[clusterName]; ok {
			return nil, status.Errorf(codes.InvalidArgument, "Cannot specify the same cluster "+
				"name with different namespace, workspace-namespace bindings are unique per "+
				"cluster per workspace.")
		}
		allClusters[clusterName] = 1
		if len(clusterName) > 0 && namespace == nil {
			return nil, status.Errorf(codes.InvalidArgument,
				"Must specify a Kubernetes namespace")
		}
		newClusterName, err := a.validateRequestClusterName(clusterName)
		if err != nil {
			return nil, err
		}
		// This might occur when using singleRM with a cluster name defined in the master config,
		// but not specified in a given request.
		if newClusterName != clusterName {
			namespaceMeta[newClusterName] = &workspacev1.WorkspaceNamespaceMeta{
				ClusterName: clusterName,
				Namespace:   namespace,
			}
			delete(namespaceMeta, clusterName)
		}
	}
	return namespaceMeta, nil
}

func validateWorkspaceName(name string) error {
	switch {
	case len(name) < 1:
		return status.Errorf(codes.InvalidArgument, "name '%s' must be at least 1 character long", name)
	case len(name) > 80:
		return status.Errorf(codes.InvalidArgument, "name '%s' must be at most 53 character long", name)
	case len(strings.TrimFunc(name, unicode.IsSpace)) == 0:
		return status.Error(codes.InvalidArgument, "name must contain at least non-whitespace letter")
	default:
		return nil
	}
}

// generateNamespaceName is the auto-generated namespace name for a given workspace-namespace
// binding in Determiend. (Note that the Kubernetes cluster name is not included in the
// name auto-generation, so auto-generated namespaces should be identical for different Kubernetes
// clusters within a given determined deployment).
// Note that Kubernetes namespace name length cannot exceed 63 characters as they must comply with
// RFC 1123 DNS labels. However, to ensure that all auto-generated namespace names are easily
// identifiable with regard to the corresponding workspace name and namespace within which the
// determined installation was created while simultaneously guaranteeing uniqueness within a given
// Kubernetes cluster, we use the following convention when auto-generating a namespace name:
// <release.namespace>(19 chars) + "_" + <cluster_id>(4) + "_" + <workspace_name>(31) + "_" +
// <workspace_id>(6).
// Note that we allow the workspace ID to occupy up to 6 characters (<= 999,999 unique
// workspaces).
func generateNamespaceName(clusterID string, workspace string, wkspID int) string {
	// Since multiple Determined installations can exist within the same Kubernetes cluster, we
	// need a readable way to distinguish auto-generated namespace names given their respective
	// determined installations. Since no two Determined installations in the same cluster should be
	// in the same namespace, we include the deployment's installation namespace as part of the
	// auto-generated namespace name.
	detInstallationNamespace := os.Getenv(kubernetesrm.ReleaseNamespaceEnvVar)
	detNamespaceCap := math.Min(float64(len(detInstallationNamespace)), float64(19))
	detNamespacePrefix := detInstallationNamespace[0:int(detNamespaceCap)]

	// In the off chance that detNamespacePrefix is identical for two different Determined
	// installations (which is possible because we cap the namespace prefix at 19 characters), use
	// the first four characters of the cluster ID, which is a less readble unique identifier that
	// we can use to further distinguish the auto-generated namespace names and lessen the
	// likelihood of an identical auto-generated namespace.
	clusterIDPrefix := clusterID[0:4]

	// Kubernetes namespaces names must follow the regex pattern [a-z0-9]([-a-z0-9]*[a-z0-9])? and
	// therefore cannot contain any capital letters or special characters other than "-".
	// When deriving part of the auto-generated namespace from the workspace name, ensure it is
	// lowercased and stripped of all characters that are out of compliance with the acceptable
	// regex pattern for namespace names.
	workspace = strings.ToLower(workspace)
	re := regexp.MustCompile(`[a-z0-9]([-a-z0-9]*[a-z0-9])?`)
	workspaceStripped := re.ReplaceAllString(workspace, "")
	workspaceCap := math.Min(float64(len(workspaceStripped)), float64(31))
	workspacePrefix := workspace[0:int(workspaceCap)]
	workspaceID := strconv.Itoa(wkspID)
	if wkspID > 999999 {
		workspaceID = workspaceID[0:6]
	}
	namespace := detNamespacePrefix + "_" + clusterIDPrefix + "_" + workspacePrefix + "_" +
		workspaceID
	return namespace
}

func (a *apiServer) GetWorkspaceByID(
	ctx context.Context, id int32, curUser model.User, rejectImmutable bool,
) (*workspacev1.Workspace, error) {
	notFoundErr := api.NotFoundErrs("workspace", strconv.Itoa(int(id)), true)
	w := &workspacev1.Workspace{}

	if err := a.m.db.QueryProto("get_workspace", w, id, curUser.ID); errors.Is(err, db.ErrNotFound) {
		return nil, notFoundErr
	} else if err != nil {
		return nil, errors.Wrapf(err, "error fetching workspace (%d) from database", id)
	}

	if err := workspace.AuthZProvider.Get().CanGetWorkspace(ctx, curUser, w); err != nil {
		return nil, authz.SubIfUnauthorized(err, notFoundErr)
	}

	if err := maskStorageConfigSecrets(w); err != nil {
		return nil, err
	}

	if rejectImmutable && w.Immutable {
		return nil, errors.Errorf("workspace (%v) is immutable and cannot add new projects", w.Id)
	}
	if rejectImmutable && w.Archived {
		return nil, errors.Errorf("workspace (%v) is archived and cannot add new projects", w.Id)
	}
	return w, nil
}

func (a *apiServer) getWorkspaceAndCheckCanDoActions(
	ctx context.Context, workspaceID int32, rejectImmutable bool,
	canDoActions ...func(context.Context, model.User, *workspacev1.Workspace) error,
) (*workspacev1.Workspace, model.User, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, model.User{}, err
	}
	w, err := a.GetWorkspaceByID(ctx, workspaceID, *curUser, rejectImmutable)
	if err != nil {
		return nil, model.User{}, err
	}

	for _, canDoAction := range canDoActions {
		if err = canDoAction(ctx, *curUser, w); err != nil {
			return nil, model.User{}, status.Error(codes.PermissionDenied, err.Error())
		}
	}
	return w, *curUser, nil
}

func (a *apiServer) workspaceHasModels(ctx context.Context, workspaceID int32) (bool, error) {
	exists, err := db.Bun().NewSelect().Table("models").
		Where("workspace_id=?", workspaceID).
		Exists(ctx)
	if err != nil {
		return false, fmt.Errorf("checking workspace for models: %w", err)
	}
	return exists, nil
}

func (a *apiServer) GetWorkspace(
	ctx context.Context, req *apiv1.GetWorkspaceRequest,
) (*apiv1.GetWorkspaceResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}

	w, err := a.GetWorkspaceByID(ctx, req.Id, *curUser, false)
	return &apiv1.GetWorkspaceResponse{Workspace: w}, err
}

func (a *apiServer) GetWorkspaceProjects(
	ctx context.Context, req *apiv1.GetWorkspaceProjectsRequest,
) (*apiv1.GetWorkspaceProjectsResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}
	if req.Id != 0 {
		if _, err = a.GetWorkspaceByID(ctx, req.Id, *curUser, false); err != nil {
			return nil, err
		}
	}

	nameFilter := req.Name
	archFilterExpr := ""
	if req.Archived != nil {
		archFilterExpr = strconv.FormatBool(req.Archived.Value)
	}
	userFilterExpr := strings.Join(req.Users, ",")
	userIds := make([]string, 0, len(req.UserIds))
	for _, userID := range req.UserIds {
		userIds = append(userIds, strconv.Itoa(int(userID)))
	}
	userIDFilterExpr := strings.Join(userIds, ",")
	// Construct the ordering expression.
	startTime := apiv1.GetWorkspaceProjectsRequest_SORT_BY_LAST_EXPERIMENT_START_TIME
	sortColMap := map[apiv1.GetWorkspaceProjectsRequest_SortBy]string{
		// `p` is an alias of `project` which is defined in master/static/srv/get_workspace_projects.sql
		apiv1.GetWorkspaceProjectsRequest_SORT_BY_UNSPECIFIED:   "p.id",
		apiv1.GetWorkspaceProjectsRequest_SORT_BY_CREATION_TIME: "p.created_at",
		startTime: "last_experiment_started_at",
		apiv1.GetWorkspaceProjectsRequest_SORT_BY_ID:          "p.id",
		apiv1.GetWorkspaceProjectsRequest_SORT_BY_NAME:        "p.name",
		apiv1.GetWorkspaceProjectsRequest_SORT_BY_DESCRIPTION: "p.description",
	}
	orderByMap := map[apiv1.OrderBy]string{
		apiv1.OrderBy_ORDER_BY_UNSPECIFIED: "ASC",
		apiv1.OrderBy_ORDER_BY_ASC:         "ASC",
		apiv1.OrderBy_ORDER_BY_DESC:        "DESC",
	}
	orderExpr := ""
	switch _, ok := sortColMap[req.SortBy]; {
	case !ok:
		return nil, fmt.Errorf("unsupported sort by %s", req.SortBy)
	case sortColMap[req.SortBy] != "id":
		orderExpr = fmt.Sprintf(
			"%s %s, id %s",
			sortColMap[req.SortBy], orderByMap[req.OrderBy], orderByMap[req.OrderBy],
		)
	default:
		orderExpr = fmt.Sprintf("id %s", orderByMap[req.OrderBy])
	}

	resp := &apiv1.GetWorkspaceProjectsResponse{}
	err = a.m.db.QueryProtof(
		"get_workspace_projects",
		[]interface{}{orderExpr},
		&resp.Projects,
		req.Id,
		userFilterExpr,
		userIDFilterExpr,
		nameFilter,
		archFilterExpr,
	)
	if err != nil {
		return nil, err
	}

	resp.Projects, err = workspace.AuthZProvider.Get().
		FilterWorkspaceProjects(ctx, *curUser, resp.Projects)
	if err != nil {
		return nil, err
	}

	return resp, api.Paginate(&resp.Pagination, &resp.Projects, req.Offset, req.Limit)
}

func (a *apiServer) GetWorkspaces(
	ctx context.Context, req *apiv1.GetWorkspacesRequest,
) (*apiv1.GetWorkspacesResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}

	nameFilter := req.Name
	nameCaseSensitiveFilter := req.NameCaseSensitive
	archFilterExpr := ""
	if req.Archived != nil {
		archFilterExpr = strconv.FormatBool(req.Archived.Value)
	}
	pinFilterExpr := ""
	if req.Pinned != nil {
		pinFilterExpr = strconv.FormatBool(req.Pinned.Value)
	}
	userFilterExpr := strings.Join(req.Users, ",")
	userIds := make([]string, 0, len(req.UserIds))
	for _, userID := range req.UserIds {
		userIds = append(userIds, strconv.Itoa(int(userID)))
	}
	userIDFilterExpr := strings.Join(userIds, ",")
	// Construct the ordering expression.
	sortColMap := map[apiv1.GetWorkspacesRequest_SortBy]string{
		apiv1.GetWorkspacesRequest_SORT_BY_UNSPECIFIED: "id",
		apiv1.GetWorkspacesRequest_SORT_BY_ID:          "id",
		apiv1.GetWorkspacesRequest_SORT_BY_NAME:        "name",
	}
	orderByMap := map[apiv1.OrderBy]string{
		apiv1.OrderBy_ORDER_BY_UNSPECIFIED: "ASC",
		apiv1.OrderBy_ORDER_BY_ASC:         "ASC",
		apiv1.OrderBy_ORDER_BY_DESC:        "DESC",
	}
	orderExpr := ""
	switch _, ok := sortColMap[req.SortBy]; {
	case !ok:
		return nil, fmt.Errorf("unsupported sort by %s", req.SortBy)
	case sortColMap[req.SortBy] != "id":
		orderExpr = fmt.Sprintf(
			"%s %s, id %s",
			sortColMap[req.SortBy], orderByMap[req.OrderBy], orderByMap[req.OrderBy],
		)
	default:
		orderExpr = fmt.Sprintf("id %s", orderByMap[req.OrderBy])
	}

	resp := &apiv1.GetWorkspacesResponse{}
	err = a.m.db.QueryProtof(
		"get_workspaces",
		[]interface{}{orderExpr},
		&resp.Workspaces,
		userFilterExpr,
		userIDFilterExpr,
		nameFilter,
		archFilterExpr,
		pinFilterExpr,
		curUser.ID,
		nameCaseSensitiveFilter,
	)
	if err != nil {
		return nil, err
	}

	resp.Workspaces, err = workspace.AuthZProvider.Get().
		FilterWorkspaces(ctx, *curUser, resp.Workspaces)
	if err != nil {
		return nil, err
	}

	return resp, api.Paginate(&resp.Pagination, &resp.Workspaces, req.Offset, req.Limit)
}

func (a *apiServer) PostWorkspace(
	ctx context.Context, req *apiv1.PostWorkspaceRequest,
) (*apiv1.PostWorkspaceResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}

	if err = workspace.AuthZProvider.Get().CanCreateWorkspace(ctx, *curUser); err != nil {
		return nil, status.Error(codes.PermissionDenied, err.Error())
	}

	if err = validateWorkspaceName(req.Name); err != nil {
		return nil, err
	}

	if req.AgentUserGroup != nil {
		err = workspace.AuthZProvider.Get().CanCreateWorkspaceWithAgentUserGroup(ctx, *curUser)
		if err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}
	}

	if req.CheckpointStorageConfig != nil && len(req.CheckpointStorageConfig.Fields) > 0 {
		if err = workspace.AuthZProvider.Get().
			CanCreateWorkspaceWithCheckpointStorageConfig(ctx, *curUser); err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}
	}

	req.ClusterNamespaceMeta, err = a.validateClusterNamespaceMeta(req.ClusterNamespaceMeta)
	if err != nil {
		return nil, err
	}

	w := &model.Workspace{
		Name: req.Name, UserID: curUser.ID,
		DefaultComputePool: req.DefaultComputePool, DefaultAuxPool: req.DefaultAuxPool,
	}

	wsnsBindings := make(map[string]*workspacev1.WorkspaceNamespaceBinding)

	err = db.Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if req.AgentUserGroup != nil {
			w.AgentUID = req.AgentUserGroup.AgentUid
			w.AgentGID = req.AgentUserGroup.AgentGid
			w.AgentUser = req.AgentUserGroup.AgentUser
			w.AgentGroup = req.AgentUserGroup.AgentGroup
		}

		if req.CheckpointStorageConfig != nil && len(req.CheckpointStorageConfig.Fields) > 0 {
			var bytes []byte
			bytes, err = req.CheckpointStorageConfig.MarshalJSON()
			if err != nil {
				return err
			}
			var sc expconf.CheckpointStorageConfig
			w.CheckpointStorageConfig = &sc
			if err = w.CheckpointStorageConfig.UnmarshalJSON(bytes); err != nil {
				return err
			}
			if err = schemas.IsComplete(w.CheckpointStorageConfig); err != nil {
				return status.Errorf(codes.InvalidArgument, err.Error())
			}
		}

		err = workspace.AddWorkspace(ctx, w, &tx)
		if err != nil {
			if strings.Contains(err.Error(), db.CodeUniqueViolation) {
				return status.Errorf(codes.AlreadyExists, "avoid names equal to other workspaces (case-insensitive)")
			}
			return err
		}

		// If the user specifies cluster-namespace pairs, send them to the request handler for adding a
		// workspace-namespace binding.
		if len(req.ClusterNamespaceMeta) > 0 {
			newReq := &apiv1.SetWorkspaceNamespaceBindingsRequest{
				WorkspaceId:          int32(w.ID),
				ClusterNamespaceMeta: req.ClusterNamespaceMeta,
			}
			wkspProto, err := w.ToProto()
			if err != nil {
				return fmt.Errorf("error converting workspace %s to proto: %w", w.Name, err)
			}
			resp, err := a.setWorkspaceNamespaceBindings(ctx, newReq, &tx, curUser, wkspProto)
			if err != nil {
				return fmt.Errorf("failed to create namespace binding: %w", err)
			}
			wsnsBindings = resp.NamespaceBindings
		}

		pin := &model.WorkspacePin{WorkspaceID: w.ID, UserID: w.UserID}
		_, err = tx.NewInsert().Model(pin).Exec(ctx)
		if err != nil {
			return errors.Wrapf(err, "error creating workspace pin %s in database", req.Name)
		}

		if err = a.AssignWorkspaceAdminToUserTx(ctx, tx, w.ID, w.UserID); err != nil {
			return errors.Wrap(err, "error assigning workspace admin")
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	protoWorkspace, err := w.ToProto()
	if err != nil {
		return nil, err
	}
	protoWorkspace.Username = curUser.Username
	protoWorkspace.Pinned = true
	return &apiv1.PostWorkspaceResponse{
		Workspace:         protoWorkspace,
		NamespaceBindings: wsnsBindings,
	}, nil
}

func (a *apiServer) DeleteWorkspaceNamespaceBindings(ctx context.Context,
	req *apiv1.DeleteWorkspaceNamespaceBindingsRequest,
) (*apiv1.DeleteWorkspaceNamespaceBindingsResponse, error) {
	currUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}
	clusterNames := []string{}
	for _, c := range req.ClusterNames {
		newClusterName, err := a.validateRequestClusterName(c)
		if err != nil {
			return nil, err
		}
		c = newClusterName
		clusterNames = append(clusterNames, c)
	}

	err = a.deleteWorkspaceNamespaceBinding(ctx, clusterNames, req.WorkspaceId, currUser)
	if err != nil {
		return nil, err
	}
	return &apiv1.DeleteWorkspaceNamespaceBindingsResponse{}, nil
}

func (a *apiServer) deleteWorkspaceNamespaceBinding(ctx context.Context,
	clusterNames []string, workspaceID int32, curUser *model.User,
) error {
	w, err := a.GetWorkspaceByID(ctx, workspaceID, *curUser, true)
	if err != nil {
		return err
	}

	if err := workspace.AuthZProvider.Get().
		CanSetWorkspaceNamespaceBindings(ctx, *curUser, w); err != nil {
		return status.Error(codes.PermissionDenied, err.Error())
	}
	if len(clusterNames) > 0 {
		err = db.Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			deletedBindings, err := workspace.DeleteWorkspaceNamespaceBindings(ctx, int(workspaceID), clusterNames, &tx)
			if err != nil {
				return err
			}
			deletedClusters := map[string]int{}
			for _, v := range deletedBindings {
				deletedClusters[v.ClusterName] = 1
				err = a.m.rm.RemoveEmptyNamespace(v.Namespace, v.ClusterName)
				if err != nil {
					return err
				}
			}
			// Since bindings to the default namespace are not stored in the db, we may
			// delete less bindings in the database than specified in the request
			if len(deletedClusters) < len(clusterNames) {
				switch {
				case len(clusterNames) == 1:
					return fmt.Errorf("tried to delete default binding for cluster %v", clusterNames[0])
				default:
					remainingClusters := ""
					for _, i := range clusterNames {
						if _, ok := deletedClusters[i]; !ok {
							remainingClusters = remainingClusters + i + ", "
						}
					}
					return fmt.Errorf("tried to delete default bindings for the following clusters: %v",
						remainingClusters[:len(remainingClusters)-2])
				}
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("failed to delete namespace binding: %w", err)
		}
	}

	return nil
}

func (a *apiServer) PatchWorkspace(
	ctx context.Context, req *apiv1.PatchWorkspaceRequest,
) (*apiv1.PatchWorkspaceResponse, error) {
	currWorkspace, currUser, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, true)
	if err != nil {
		return nil, err
	}

	insertColumns := []string{}
	updatedWorkspace := model.Workspace{}

	if req.Workspace.Name != nil {
		err = validateWorkspaceName(req.Workspace.Name.Value)
		if err != nil {
			return nil, err
		}
	}

	if req.Workspace.Name != nil && req.Workspace.Name.Value != currWorkspace.Name {
		if err = workspace.AuthZProvider.Get().
			CanSetWorkspacesName(ctx, currUser, currWorkspace); err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}

		log.Infof("workspace (%d) name changing from \"%s\" to \"%s\"",
			currWorkspace.Id, currWorkspace.Name, req.Workspace.Name.Value)
		insertColumns = append(insertColumns, "name")
		updatedWorkspace.Name = req.Workspace.Name.Value
	}

	if req.Workspace.AgentUserGroup != nil {
		if err = workspace.AuthZProvider.Get().
			CanSetWorkspacesAgentUserGroup(ctx, currUser, currWorkspace); err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}

		updateAug := req.Workspace.AgentUserGroup

		updatedWorkspace.AgentUID = updateAug.AgentUid
		updatedWorkspace.AgentGID = updateAug.AgentGid
		updatedWorkspace.AgentUser = updateAug.AgentUser
		updatedWorkspace.AgentGroup = updateAug.AgentGroup

		insertColumns = append(insertColumns, "uid", "user_", "gid", "group_")
	}

	if req.Workspace.DefaultComputeResourcePool != nil ||
		req.Workspace.DefaultAuxResourcePool != nil {
		if err = workspace.AuthZProvider.Get().
			CanSetWorkspacesDefaultPools(ctx, currUser, currWorkspace); err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}

		rpConfigs, err := a.resourcePoolsAsConfigs()
		if err != nil {
			return nil, err
		}
		rpNamesSlice, _, err := db.ReadRPsAvailableToWorkspace(
			ctx, currWorkspace.Id, 0, -1, rpConfigs,
		)
		if err != nil {
			return nil, err
		}

		rpNames := set.FromSlice(append(rpNamesSlice, ""))

		if req.Workspace.DefaultComputeResourcePool != nil {
			if !rpNames.Contains(*req.Workspace.DefaultComputeResourcePool) {
				return nil, status.Error(codes.FailedPrecondition, "unable to bind a resource "+
					"pool that does not exist or is not available to the workspace")
			}
			updatedWorkspace.DefaultComputePool = *req.Workspace.DefaultComputeResourcePool
			insertColumns = append(insertColumns, "default_compute_pool")
		}
		if req.Workspace.DefaultAuxResourcePool != nil {
			if !rpNames.Contains(*req.Workspace.DefaultAuxResourcePool) {
				return nil, status.Error(codes.FailedPrecondition, "unable to bind a resource "+
					"pool that does not exist or is not available to the workspace")
			}
			updatedWorkspace.DefaultAuxPool = *req.Workspace.DefaultAuxResourcePool
			insertColumns = append(insertColumns, "default_aux_pool")
		}
	}

	if req.Workspace.CheckpointStorageConfig != nil {
		if err = workspace.AuthZProvider.Get().
			CanSetWorkspacesCheckpointStorageConfig(ctx, currUser, currWorkspace); err != nil {
			return nil, status.Error(codes.PermissionDenied, err.Error())
		}

		if len(req.Workspace.CheckpointStorageConfig.Fields) > 0 {
			var bytes []byte
			bytes, err = req.Workspace.CheckpointStorageConfig.MarshalJSON()
			if err != nil {
				return nil, err
			}
			var sc expconf.CheckpointStorageConfig
			updatedWorkspace.CheckpointStorageConfig = &sc
			if err = updatedWorkspace.CheckpointStorageConfig.UnmarshalJSON(bytes); err != nil {
				return nil, err
			}
			if err = schemas.IsComplete(updatedWorkspace.CheckpointStorageConfig); err != nil {
				return nil, status.Errorf(codes.InvalidArgument, err.Error())
			}
		}
		insertColumns = append(insertColumns, "checkpoint_storage_config")

		toDelete := []string{}
		for cluster, metadata := range req.Workspace.ClusterNamespaceMeta {
			if metadata.Namespace == nil {
				toDelete = append(toDelete, cluster)
				delete(req.Workspace.ClusterNamespaceMeta, cluster)
			}
		}
		if len(toDelete) > 0 {
			err = a.deleteWorkspaceNamespaceBinding(ctx, toDelete, req.Id, &currUser)
			if err != nil && !strings.Contains(err.Error(), "tried to delete default binding.") {
				return nil, err
			}
		}

		if len(req.Workspace.ClusterNamespaceMeta) != 0 {
			newReq := &apiv1.SetWorkspaceNamespaceBindingsRequest{
				WorkspaceId:          req.Id,
				ClusterNamespaceMeta: req.Workspace.ClusterNamespaceMeta,
			}
			err = db.Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
				_, err := a.setWorkspaceNamespaceBindings(ctx, newReq, &tx, &currUser, currWorkspace)
				if err != nil {
					return fmt.Errorf("failed to create namespace binding: %w", err)
				}
				return nil
			})
			if err != nil {
				return nil, err
			}
		}
	}

	if len(insertColumns) == 0 {
		return &apiv1.PatchWorkspaceResponse{Workspace: currWorkspace}, nil
	}

	_, err = db.Bun().NewUpdate().Model(&updatedWorkspace).
		Column(insertColumns...).
		Where("id = ?", currWorkspace.Id).
		Exec(ctx)
	if err != nil {
		if strings.Contains(err.Error(), db.CodeUniqueViolation) {
			return nil,
				status.Errorf(codes.AlreadyExists, "avoid names equal to other workspaces (case-insensitive)")
		}
		return nil, err
	}

	// TODO(ilia): Avoid second refetch.
	finalWorkspace, err := a.GetWorkspaceByID(ctx, currWorkspace.Id, currUser, false)
	return &apiv1.PatchWorkspaceResponse{Workspace: finalWorkspace},
		errors.Wrapf(err, "error refetching updated workspace (%d) from db", currWorkspace.Id)
}

func (a *apiServer) setWorkspaceNamespaceBindings(ctx context.Context,
	req *apiv1.SetWorkspaceNamespaceBindingsRequest, tx *bun.Tx, curUser *model.User,
	w *workspacev1.Workspace,
) (*apiv1.SetWorkspaceNamespaceBindingsResponse, error) {
	// TODO(amanda): When we land auto-creating namespaces, add an AND to this if condition checking
	// for nil namespace name AND False auto-create namespace.
	// NamespaceName is an optional param in the request's proto message to leave room for the
	// auto-create namespace feature.
	wkspID := int(w.Id)
	if err := workspace.AuthZProvider.Get().
		CanSetWorkspaceNamespaceBindings(ctx, *curUser, w); err != nil {
		return nil, status.Error(codes.PermissionDenied, err.Error())
	}

	namespaceBindings := make(map[string]*workspacev1.WorkspaceNamespaceBinding)
	for clusterName, namespaceMetadata := range req.ClusterNamespaceMeta {
		namespace := namespaceMetadata.Namespace
		// Verify that the namespace exists in the Kubernetes cluster for the corresponding RM.
		err := a.m.rm.VerifyNamespaceExists(*namespace, clusterName)
		if err != nil {
			return nil, fmt.Errorf("error verifying Kubernetes namespace: %w", err)
		}

		// Since workspace-namespace bindings for the default namespace of a given cluster are not
		// automatically saved in the database, maintain constistency by not saving default
		// namespace bindings to the db if a user tries to set them.
		defaultNamespace, err := a.m.rm.DefaultNamespace(clusterName)
		if err != nil {
			return nil, err
		}
		isDefaultNamespace := namespace == defaultNamespace

		// Change the workspace-namespace binding for the given cluster if it already exists. If no
		// namespace is bound to the workspace for the specified cluster, add one.
		findWsNsQuery := tx.NewSelect().Model(&model.WorkspaceNamespace{}).
			Where("workspace_id = ?", wkspID).
			Where("cluster_name = ?", clusterName)

		var wsns model.WorkspaceNamespace
		err = findWsNsQuery.Scan(ctx, &wsns)
		if err != nil {
			if err != sql.ErrNoRows {
				return nil, fmt.Errorf("error getting the current workspace-namespace binding: %w", err)
			}
			namespaceBindings[clusterName] = &workspacev1.WorkspaceNamespaceBinding{
				WorkspaceId: int32(wkspID),
				Namespace:   *namespace,
				ClusterName: clusterName,
			}
			// The workspace has no namespace binding for the specified cluster, so we add one.
			wsns = model.WorkspaceNamespace{
				WorkspaceID: wkspID,
				Namespace:   *namespace,
				ClusterName: clusterName,
			}
			if !isDefaultNamespace {
				err = workspace.AddWorkspaceNamespaceBinding(ctx, &wsns, tx)
				if err != nil {
					return nil, err
				}
			}
		} else if wsns.Namespace != *namespace {
			if !isDefaultNamespace {
				// Update the existing workspace-namespace binding for the specified cluster.
				_, err = tx.NewUpdate().Model(&model.WorkspaceNamespace{}).
					Set("namespace = ?", namespace).
					Where("workspace_id = ?", wkspID).
					Where("namespace = ?", wsns.Namespace).
					Where("cluster_name = ?", clusterName).
					Exec(ctx)
				if err != nil {
					return nil, fmt.Errorf("could not update workspace-namespace binding: %w", err)
				}
			} else {
				// Delete the existing workspace-namespace binding if a user wants to bind the
				// workspace to the default namespace for a given cluster.
				deletedBindings, err := workspace.DeleteWorkspaceNamespaceBindings(ctx, wsns.WorkspaceID, []string{clusterName}, tx)
				if err != nil {
					return nil, err
				}
				for _, v := range deletedBindings {
					err = a.m.rm.RemoveEmptyNamespace(v.Namespace, v.ClusterName)
					if err != nil {
						return nil, err
					}
				}
			}
		}
		namespaceBindings[clusterName] = &workspacev1.WorkspaceNamespaceBinding{
			WorkspaceId: w.Id,
			Namespace:   *namespace, ClusterName: clusterName,
		}
	}
	resp := &apiv1.SetWorkspaceNamespaceBindingsResponse{NamespaceBindings: namespaceBindings}
	return resp, nil
}

func (a *apiServer) SetWorkspaceNamespaceBindings(ctx context.Context,
	req *apiv1.SetWorkspaceNamespaceBindingsRequest,
) (*apiv1.SetWorkspaceNamespaceBindingsResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}

	w, err := a.GetWorkspaceByID(ctx, req.WorkspaceId, *curUser, false)
	if err != nil {
		return nil, err
	}

	req.ClusterNamespaceMeta, err = a.validateClusterNamespaceMeta(req.ClusterNamespaceMeta)
	if err != nil {
		return nil, err
	}

	var finalRes *apiv1.SetWorkspaceNamespaceBindingsResponse
	err = db.Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		res, err := a.setWorkspaceNamespaceBindings(ctx, req, &tx, curUser, w)
		if err != nil {
			return err
		}
		finalRes = res
		return nil
	})
	return finalRes, err
}

func (a *apiServer) deleteWorkspace(
	ctx context.Context, workspaceID int32, projects []*projectv1.Project,
) {
	log.Debugf("deleting workspace %d projects", workspaceID)
	holder := &workspacev1.Workspace{}
	for _, pj := range projects {
		expList, err := db.ProjectExperiments(context.TODO(), int(pj.Id))
		if err != nil {
			log.WithError(err).Errorf("error fetching experiments on project %d while deleting workspace %d",
				pj.Id, workspaceID)
			_ = a.m.db.QueryProto("delete_fail_workspace", holder, workspaceID, err.Error())
			return
		}
		err = a.deleteProject(ctx, pj.Id, expList)
		if err != nil {
			log.WithError(err).Errorf("error deleting project %d while deleting workspace %d", pj.Id,
				workspaceID)
			_ = a.m.db.QueryProto("delete_fail_workspace", holder, workspaceID, err.Error())
			return
		}
	}

	err := a.m.db.QueryProto("delete_workspace", holder, workspaceID)
	if err != nil {
		log.WithError(err).Errorf("failed to delete workspace %d", workspaceID)
		_ = a.m.db.QueryProto("delete_fail_workspace", holder, workspaceID, err.Error())
		return
	}
	log.Debugf("workspace %d deleted successfully", workspaceID)
}

func (a *apiServer) ListWorkspaceNamespaceBindings(
	ctx context.Context,
	req *apiv1.ListWorkspaceNamespaceBindingsRequest,
) (*apiv1.ListWorkspaceNamespaceBindingsResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}

	_, err = a.GetWorkspaceByID(ctx, req.Id, *curUser, false)
	if err != nil {
		return nil, err
	}

	protoWorkspaceNamespaceBindings := make(map[string]string)

	// If a workspce does not have a workspace-namespace binding for a given cluster in the
	// database, it is bound to the either the namespace specified for the corresponding resource
	// manager in the master config or to the default Kubernetes namespace is the former field is
	// empty.
	for rmClusterName := range a.m.allRms {
		defaultNamespace, err := a.m.rm.DefaultNamespace(rmClusterName)
		if err != nil {
			return nil, fmt.Errorf("error getting default namespace: %w", err)
		}
		var wsns model.WorkspaceNamespace
		protoBinding := &workspacev1.WorkspaceNamespaceBinding{
			WorkspaceId: req.Id,
			Namespace:   *defaultNamespace,
			ClusterName: rmClusterName,
		}

		err = db.Bun().NewSelect().
			Model(&model.WorkspaceNamespace{}).
			Where("workspace_id = ?", req.Id).
			Where("cluster_name LIKE ?", rmClusterName).
			Scan(ctx, &wsns)
		if err != nil && err != sql.ErrNoRows {
			return nil, fmt.Errorf("error getting workspace-namespace bindings: %w", err)
		}
		if err == nil {
			protoBinding = wsns.ToProto()
		}
		protoWorkspaceNamespaceBindings[protoBinding.ClusterName] = protoBinding.Namespace
	}

	// A workspace-namespace binding is considered "stale" if its cluster name is not used by any
	// of the resource managers for the given determined deployment. List stale bindings as well,
	// labeling them as such.
	workspaceNamespaceBindings, err := workspace.GetWorkspaceNamespaceBindings(ctx, int(req.Id))
	if err != nil {
		return nil, fmt.Errorf("error getting workspace-namespace bindings: %w", err)
	}
	for _, binding := range workspaceNamespaceBindings {
		protoBinding := binding.ToProto()
		if _, ok := a.m.allRms[binding.ClusterName]; !ok {
			log.Warnf("Cluster with name %s is stale", protoBinding.ClusterName)
			// Edge case where a workspace-namespace binding has a cluster name identical to a
			// cluster name with a stale label. In these cases, we should return the current
			// active cluster (whose name coincidentially includes a stale label).
			_, ok = protoWorkspaceNamespaceBindings[protoBinding.ClusterName+staleLabel]
			if !ok {
				protoBinding.ClusterName += staleLabel
				protoWorkspaceNamespaceBindings[protoBinding.ClusterName] = protoBinding.Namespace
			}
		}
	}

	return &apiv1.ListWorkspaceNamespaceBindingsResponse{
		NamespaceBindings: protoWorkspaceNamespaceBindings,
	}, nil
}

func (a *apiServer) DeleteWorkspace(
	ctx context.Context,
	req *apiv1.DeleteWorkspaceRequest,
) (*apiv1.DeleteWorkspaceResponse, error) {
	_, _, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, false,
		workspace.AuthZProvider.Get().CanDeleteWorkspace)
	if err != nil {
		return nil, err
	}

	modelsExist, err := a.workspaceHasModels(ctx, req.Id)
	if err != nil {
		return nil, err
	}
	if modelsExist {
		return nil, status.Errorf(codes.FailedPrecondition, "workspace (%d) contains models; move or delete models first",
			req.Id)
	}

	holder := &workspacev1.Workspace{}
	// TODO(kristine): DET-10138 update workspace state in transaction with template delete
	err = a.m.db.QueryProto("deletable_workspace", holder, req.Id)
	if err != nil || holder.Id == 0 {
		return nil, fmt.Errorf("workspace (%d) does not exist or not deletable by this user: %w", req.Id, err)
	}

	projects := []*projectv1.Project{}
	err = a.m.db.QueryProtof(
		"get_workspace_projects",
		[]interface{}{"id ASC"},
		&projects,
		req.Id,
		"",
		"",
		"",
		"",
	)
	if err != nil {
		return nil, fmt.Errorf("getting workspace projects: %w", err)
	}

	log.Debugf("deleting workspace %d NTSC", req.Id)
	command.DefaultCmdService.DeleteWorkspaceNTSC(req)

	log.Debugf("deleting workspace %d templates", req.Id)
	err = templates.DeleteWorkspaceTemplates(ctx, int(req.Id))
	if err != nil {
		return nil, fmt.Errorf("error deleting workspace (%d) templates: %w", req.Id, err)
	}
	// get the workspace-namespace bindings associated with the workspace.
	var toBeDeletedBindings []model.WorkspaceNamespace
	err = db.Bun().NewSelect().
		Model(&model.WorkspaceNamespace{}).
		Where("workspace_id = ?", req.Id).
		Scan(ctx, &toBeDeletedBindings)
	if err != nil {
		return nil, fmt.Errorf("error getting workspace-namespace bindings")
	}

	// Get the auto-generated namespace name for the workspace.
	autoNamespaceName, err := getAutoGeneratedNamespaceName(ctx, int(req.Id))
	if err != nil {
		return nil, fmt.Errorf("error getting auto-generated namespace: %w", err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		a.deleteWorkspace(ctx, req.Id, projects)
	}()
	wg.Wait()

	// Delete the auto-generated namespace (if it exists) and its resources in Kubernetes.
	err = a.m.rm.DeleteNamespace(*autoNamespaceName)
	if err != nil {
		return nil, err
	}
	for _, v := range toBeDeletedBindings {
		// Only remove the namespace if binding is not stale
		if _, ok := a.m.allRms[v.ClusterName]; ok {
			err = a.m.rm.RemoveEmptyNamespace(v.Namespace, v.ClusterName)
			if err != nil {
				return nil, err
			}
		}
	}

	if len(projects) == 0 {
		return &apiv1.DeleteWorkspaceResponse{Completed: true}, nil
	}
	return &apiv1.DeleteWorkspaceResponse{Completed: false}, nil
}

func getAutoGeneratedNamespaceName(ctx context.Context, wkspID int) (*string, error) {
	var wsToDelete model.Workspace
	err := db.Bun().NewSelect().Model(&model.Workspace{}).Where("id = ?", wkspID).Scan(ctx,
		&wsToDelete)
	if err != nil {
		return nil, fmt.Errorf("error getting workspace %d: %w", wkspID, err)
	}

	var clusterID string
	err = db.Bun().NewSelect().Table("cluster_id").Column("cluster_id").Scan(ctx, &clusterID)
	if err != nil {
		return nil, fmt.Errorf("error getting cluster ID: %w", err)
	}

	namespace := generateNamespaceName(clusterID, wsToDelete.Name, wsToDelete.ID)
	return &namespace, nil
}

func (a *apiServer) ArchiveWorkspace(
	ctx context.Context, req *apiv1.ArchiveWorkspaceRequest) (*apiv1.ArchiveWorkspaceResponse,
	error,
) {
	_, _, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, false,
		workspace.AuthZProvider.Get().CanArchiveWorkspace)
	if err != nil {
		return nil, err
	}

	holder := &workspacev1.Workspace{}
	if err = a.m.db.QueryProto("archive_workspace", holder, req.Id, true); err != nil {
		return nil, errors.Wrapf(err, "error archiving workspace (%d)", req.Id)
	}
	if holder.Id == 0 {
		return nil, errors.Wrapf(err, "workspace (%d) does not exist or not archive-able by this user",
			req.Id)
	}
	return &apiv1.ArchiveWorkspaceResponse{}, nil
}

func (a *apiServer) UnarchiveWorkspace(
	ctx context.Context, req *apiv1.UnarchiveWorkspaceRequest) (*apiv1.UnarchiveWorkspaceResponse,
	error,
) {
	_, _, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, false,
		workspace.AuthZProvider.Get().CanUnarchiveWorkspace)
	if err != nil {
		return nil, err
	}

	holder := &workspacev1.Workspace{}
	if err = a.m.db.QueryProto("archive_workspace", holder, req.Id, false); err != nil {
		return nil, errors.Wrapf(err, "error unarchiving workspace (%d)", req.Id)
	}
	if holder.Id == 0 {
		return nil, errors.Wrapf(err,
			"workspace (%d) does not exist or not unarchive-able by this user", req.Id)
	}
	return &apiv1.UnarchiveWorkspaceResponse{}, nil
}

func (a *apiServer) PinWorkspace(
	ctx context.Context, req *apiv1.PinWorkspaceRequest,
) (*apiv1.PinWorkspaceResponse, error) {
	_, currUser, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, false,
		workspace.AuthZProvider.Get().CanPinWorkspace)
	if err != nil {
		return nil, err
	}

	err = a.m.db.QueryProto("pin_workspace", &workspacev1.Workspace{}, req.Id, currUser.ID)

	return &apiv1.PinWorkspaceResponse{},
		errors.Wrapf(err, "error pinning workspace (%d)", req.Id)
}

func (a *apiServer) UnpinWorkspace(
	ctx context.Context, req *apiv1.UnpinWorkspaceRequest,
) (*apiv1.UnpinWorkspaceResponse, error) {
	_, currUser, err := a.getWorkspaceAndCheckCanDoActions(ctx, req.Id, false,
		workspace.AuthZProvider.Get().CanUnpinWorkspace)
	if err != nil {
		return nil, err
	}

	err = a.m.db.QueryProto("unpin_workspace", &workspacev1.Workspace{}, req.Id, currUser.ID)

	return &apiv1.UnpinWorkspaceResponse{},
		errors.Wrapf(err, "error un-pinning workspace (%d)", req.Id)
}

func (a *apiServer) ListRPsBoundToWorkspace(
	ctx context.Context, req *apiv1.ListRPsBoundToWorkspaceRequest,
) (*apiv1.ListRPsBoundToWorkspaceResponse, error) {
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}
	err = workspace.AuthZProvider.Get().CanGetWorkspaceID(
		ctx, *curUser, req.WorkspaceId,
	)
	if err != nil {
		return nil, err
	}

	rpConfigs, err := a.resourcePoolsAsConfigs()
	if err != nil {
		return nil, err
	}
	rpNames, pagination, err := db.ReadRPsAvailableToWorkspace(
		ctx, req.WorkspaceId, req.Offset, req.Limit, rpConfigs,
	)
	if err != nil {
		return nil, err
	}

	return &apiv1.ListRPsBoundToWorkspaceResponse{
		ResourcePools: rpNames,
		Pagination:    pagination,
	}, nil
}
