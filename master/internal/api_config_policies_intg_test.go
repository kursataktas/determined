//go:build integration
// +build integration

package internal

import (
	"database/sql"
	"fmt"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/determined-ai/determined/master/internal/configpolicy"
	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/internal/mocks"
	"github.com/determined-ai/determined/master/internal/user"
	"github.com/determined-ai/determined/master/internal/workspace"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/ptrs"
	"github.com/determined-ai/determined/master/test/testutils"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
)

func TestDeleteWorkspaceConfigPolicies(t *testing.T) {
	// TODO (CM-520): Make test cases for experiment config policies.

	// Create one workspace and continuously set and delete config policies from there
	api, curUser, ctx := setupAPITest(t, nil)
	testutils.MustLoadLicenseAndKeyFromFilesystem("../../")

	wkspResp, err := api.PostWorkspace(ctx, &apiv1.PostWorkspaceRequest{Name: uuid.New().String()})
	require.NoError(t, err)
	workspaceID := wkspResp.Workspace.Id
	cases := []struct {
		name string
		req  *apiv1.DeleteWorkspaceConfigPoliciesRequest
		err  error
	}{
		{
			"invalid workload type",
			&apiv1.DeleteWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID,
				WorkloadType: "bad workload type",
			},
			fmt.Errorf("invalid workload type"),
		},
		{
			"empty workload type",
			&apiv1.DeleteWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID,
				WorkloadType: "",
			},
			fmt.Errorf(noWorkloadErr),
		},
		{
			"valid request",
			&apiv1.DeleteWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID,
				WorkloadType: model.NTSCType,
			},
			nil,
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			ntscPolicies := &model.TaskConfigPolicies{
				WorkspaceID:   ptrs.Ptr(int(test.req.WorkspaceId)),
				WorkloadType:  model.NTSCType,
				LastUpdatedBy: curUser.ID,
			}
			err = configpolicy.SetTaskConfigPolicies(ctx, ntscPolicies)
			require.NoError(t, err)

			resp, err := api.DeleteWorkspaceConfigPolicies(ctx, test.req)
			if test.err != nil {
				require.ErrorContains(t, err, test.err.Error())
				return
			}
			// Delete successful?
			require.NoError(t, err)
			require.NotNil(t, resp)

			// Policies removed?
			policies, err := configpolicy.GetTaskConfigPolicies(ctx, ptrs.Ptr(int(workspaceID)), test.req.WorkloadType)
			require.Nil(t, policies)
			require.ErrorIs(t, err, sql.ErrNoRows)
		})
	}

	// Test invalid workspace ID.
	resp, err := api.DeleteWorkspaceConfigPolicies(ctx, &apiv1.DeleteWorkspaceConfigPoliciesRequest{
		WorkspaceId:  -1,
		WorkloadType: model.NTSCType,
	})
	require.Nil(t, resp)
	require.ErrorContains(t, err, "not found")
}

func TestDeleteGlobalConfigPolicies(t *testing.T) {
	// TODO (CM-520): Make test cases for experiment config policies.

	api, curUser, ctx := setupAPITest(t, nil)
	testutils.MustLoadLicenseAndKeyFromFilesystem("../../")

	cases := []struct {
		name string
		req  *apiv1.DeleteGlobalConfigPoliciesRequest
		err  error
	}{
		{
			"invalid workload type",
			&apiv1.DeleteGlobalConfigPoliciesRequest{
				WorkloadType: "invalid workload type",
			},
			fmt.Errorf("invalid workload type"),
		},
		{
			"empty workload type",
			&apiv1.DeleteGlobalConfigPoliciesRequest{
				WorkloadType: "",
			},
			fmt.Errorf(noWorkloadErr),
		},
		{
			"valid request",
			&apiv1.DeleteGlobalConfigPoliciesRequest{
				WorkloadType: model.NTSCType,
			},
			nil,
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := configpolicy.SetTaskConfigPolicies(ctx, &model.TaskConfigPolicies{
				WorkloadType:  model.NTSCType,
				LastUpdatedBy: curUser.ID,
			})
			require.NoError(t, err)

			resp, err := api.DeleteGlobalConfigPolicies(ctx, test.req)
			if test.err != nil {
				require.ErrorContains(t, err, test.err.Error())
				return
			}
			// Delete successful?
			require.NoError(t, err)
			require.NotNil(t, resp)

			// Policies removed?
			policies, err := configpolicy.GetTaskConfigPolicies(ctx, nil, test.req.WorkloadType)
			require.Nil(t, policies)
			require.ErrorIs(t, err, sql.ErrNoRows)
		})
	}
}

func TestBasicRBACConfigPolicyPerms(t *testing.T) {
	api, curUser, ctx := setupAPITest(t, nil)
	curUser.Admin = false
	err := user.Update(ctx, &curUser, []string{"admin"}, nil)
	require.NoError(t, err)

	testutils.MustLoadLicenseAndKeyFromFilesystem("../../")

	resp, err := api.PostWorkspace(ctx, &apiv1.PostWorkspaceRequest{Name: uuid.New().String()})
	require.NoError(t, err)
	wkspID := resp.Workspace.Id

	wksp, err := workspace.WorkspaceByName(ctx, resp.Workspace.Name)
	require.NoError(t, err)
	newUser, err := db.HackAddUser(ctx, &model.User{Username: uuid.NewString()})
	require.NoError(t, err)

	wksp.UserID = newUser
	_, err = db.Bun().NewUpdate().Model(wksp).Where("id = ?", wksp.ID).Exec(ctx)
	require.NoError(t, err)

	cases := []struct {
		name string
		req  func() error
		err  error
	}{
		{
			"delete workspace config policies",
			func() error {
				_, err := api.DeleteWorkspaceConfigPolicies(ctx,
					&apiv1.DeleteWorkspaceConfigPoliciesRequest{
						WorkspaceId:  wkspID,
						WorkloadType: model.NTSCType,
					},
				)
				return err
			},
			fmt.Errorf("only admins may set config policies for workspaces"),
		},
		{
			"delete global config policies",
			func() error {
				_, err := api.DeleteGlobalConfigPolicies(ctx,
					&apiv1.DeleteGlobalConfigPoliciesRequest{
						WorkloadType: model.NTSCType,
					},
				)
				return err
			},
			fmt.Errorf("PermissionDenied"),
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := test.req()
			require.ErrorContains(t, err, test.err.Error())
		})
	}
}

func TestGetWorkspaceConfigPolicies(t *testing.T) {
	api, curUser, ctx := setupAPITest(t, nil)
	testutils.MustLoadLicenseAndKeyFromFilesystem("../../")

	wkspResp, err := api.PostWorkspace(ctx, &apiv1.PostWorkspaceRequest{Name: uuid.New().String()})
	require.NoError(t, err)
	workspaceID1 := wkspResp.Workspace.Id
	wkspResp, err = api.PostWorkspace(ctx, &apiv1.PostWorkspaceRequest{Name: uuid.New().String()})
	require.NoError(t, err)
	workspaceID2 := wkspResp.Workspace.Id

	// set only config policy
	taskConfigPolicies := &model.TaskConfigPolicies{
		WorkspaceID:     ptrs.Ptr(int(workspaceID1)),
		WorkloadType:    model.NTSCType,
		LastUpdatedBy:   curUser.ID,
		InvariantConfig: ptrs.Ptr(configpolicy.DefaultInvariantConfigStr),
	}
	err = configpolicy.SetTaskConfigPolicies(ctx, taskConfigPolicies)
	require.NoError(t, err)

	// set only constraints policy
	taskConfigPolicies = &model.TaskConfigPolicies{
		WorkspaceID:   ptrs.Ptr(int(workspaceID1)),
		WorkloadType:  model.ExperimentType,
		LastUpdatedBy: curUser.ID,
		Constraints:   ptrs.Ptr(configpolicy.DefaultConstraintsStr),
	}
	err = configpolicy.SetTaskConfigPolicies(ctx, taskConfigPolicies)
	require.NoError(t, err)

	// set both config and constraints policy
	taskConfigPolicies = &model.TaskConfigPolicies{
		WorkspaceID:     ptrs.Ptr(int(workspaceID2)),
		WorkloadType:    model.NTSCType,
		LastUpdatedBy:   curUser.ID,
		InvariantConfig: ptrs.Ptr(configpolicy.DefaultInvariantConfigStr),
		Constraints:     ptrs.Ptr(configpolicy.DefaultConstraintsStr),
	}
	err = configpolicy.SetTaskConfigPolicies(ctx, taskConfigPolicies)
	require.NoError(t, err)

	cases := []struct {
		name           string
		req            *apiv1.GetWorkspaceConfigPoliciesRequest
		err            error
		hasConfig      bool
		hasConstraints bool
	}{
		{
			"invalid workload type",
			&apiv1.GetWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID1,
				WorkloadType: "bad workload type",
			},
			fmt.Errorf("invalid workload type"),
			false,
			false,
		},
		{
			"empty workload type",
			&apiv1.GetWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID1,
				WorkloadType: "",
			},
			fmt.Errorf(noWorkloadErr),
			false,
			false,
		},
		{
			"valid request only config",
			&apiv1.GetWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID1,
				WorkloadType: model.NTSCType,
			},
			nil,
			true,
			false,
		},
		{
			"valid request only constraints",
			&apiv1.GetWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID1,
				WorkloadType: model.ExperimentType,
			},
			nil,
			false,
			true,
		},
		{
			"valid request both configs and constraints",
			&apiv1.GetWorkspaceConfigPoliciesRequest{
				WorkspaceId:  workspaceID2,
				WorkloadType: model.NTSCType,
			},
			nil,
			true,
			true,
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			resp, err := api.GetWorkspaceConfigPolicies(ctx, test.req)
			if test.err != nil {
				require.ErrorContains(t, err, test.err.Error())
				return
			}
			require.NoError(t, err)
			require.NotNil(t, resp)

			if test.hasConfig {
				require.Contains(t, resp.ConfigPolicies.String(), "config")
			} else {
				require.NotContains(t, resp.ConfigPolicies.String(), "config")
			}

			if test.hasConstraints {
				require.Contains(t, resp.ConfigPolicies.String(), "constraints")
			} else {
				require.NotContains(t, resp.ConfigPolicies.String(), "constraints")
			}
		})
	}
}

func TestAuthZCanModifyConfigPolicies(t *testing.T) {
	api, workspaceAuthZ, _, ctx := setupWorkspaceAuthZTest(t, nil)
	testutils.MustLoadLicenseAndKeyFromFilesystem("../../")
	configPolicyAuthZ := setupConfigPolicyAuthZ()

	workspaceAuthZ.On("CanCreateWorkspace", mock.Anything, mock.Anything).Return(nil)
	workspaceAuthZ.On("CanGetWorkspace", mock.Anything, mock.Anything, mock.Anything).
		Return(nil).Once()

	wkspResp, err := api.PostWorkspace(ctx, &apiv1.PostWorkspaceRequest{Name: uuid.New().String()})
	require.NoError(t, err)
	workspaceID := wkspResp.Workspace.Id

	// (Workspace-level) Deny with permission access error.
	expectedErr := fmt.Errorf("canModifyConfigPoliciesError")
	configPolicyAuthZ.On("CanModifyWorkspaceConfigPolicies", mock.Anything, mock.Anything,
		mock.Anything).Return(expectedErr).Once()

	_, err = api.DeleteWorkspaceConfigPolicies(ctx,
		&apiv1.DeleteWorkspaceConfigPoliciesRequest{
			WorkspaceId:  workspaceID,
			WorkloadType: model.NTSCType,
		})
	require.Equal(t, expectedErr, err)

	// Nil error returns whatever the delete request returned.
	configPolicyAuthZ.On("CanModifyWorkspaceConfigPolicies", mock.Anything, mock.Anything,
		mock.Anything).Return(nil).Once()
	_, err = api.DeleteWorkspaceConfigPolicies(ctx,
		&apiv1.DeleteWorkspaceConfigPoliciesRequest{
			WorkspaceId:  workspaceID,
			WorkloadType: model.NTSCType,
		})
	require.NoError(t, err)

	workspaceAuthZ.On("CanCreateWorkspace", mock.Anything, mock.Anything).Return(nil)

	// (Global) Deny with permission access error.
	expectedErr = fmt.Errorf("canModifyGlobalConfigPoliciesError")
	configPolicyAuthZ.On("CanModifyGlobalConfigPolicies", mock.Anything, mock.Anything).
		Return(expectedErr, nil).Once()

	_, err = api.DeleteGlobalConfigPolicies(ctx,
		&apiv1.DeleteGlobalConfigPoliciesRequest{WorkloadType: model.NTSCType})
	require.Equal(t, expectedErr, err)

	// Nil error returns whatever the delete request returned.
	configPolicyAuthZ.On("CanModifyGlobalConfigPolicies", mock.Anything, mock.Anything).
		Return(nil, nil).Once()
	_, err = api.DeleteGlobalConfigPolicies(ctx,
		&apiv1.DeleteGlobalConfigPoliciesRequest{WorkloadType: model.NTSCType})
	require.NoError(t, err)
}

var cpAuthZ *mocks.ConfigPolicyAuthZ

func setupConfigPolicyAuthZ() *mocks.ConfigPolicyAuthZ {
	if cpAuthZ == nil {
		cpAuthZ = &mocks.ConfigPolicyAuthZ{}
		configpolicy.AuthZProvider.Register("mock", cpAuthZ)
	}
	return cpAuthZ
}
