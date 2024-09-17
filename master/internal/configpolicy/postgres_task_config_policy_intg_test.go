//go:build integration
// +build integration

package configpolicy

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	log "github.com/sirupsen/logrus"

	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/pkg/etc"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/ptrs"

	"github.com/stretchr/testify/require"
)

func TestSetNTSCConfigPolicies(t *testing.T) {
	ctx := context.Background()
	require.NoError(t, etc.SetRootPath(db.RootFromDB))
	pgDB, cleanup := db.MustResolveNewPostgresDatabase(t)
	defer cleanup()
	db.MustMigrateTestPostgres(t, pgDB, db.MigrationsFromDB)

	user := db.RequireMockUser(t, pgDB)
	workspaceIDs := []int32{}

	defer func() {
		if len(workspaceIDs) > 0 {
			err := db.CleanupMockWorkspace(workspaceIDs)
			if err != nil {
				log.Errorf("error when cleaning up mock workspaces")
			}
		}
	}()

	tests := []struct {
		name     string
		ntscTCPs *model.NTSCTaskConfigPolicies
		global   bool
		err      *string
	}{
		{
			"invalid user id",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   -1,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: DefaultCommandConfig(),
				Constraints:     DefaultConstraints(),
			},
			false,
			ptrs.Ptr("violates foreign key constraint"),
		},
		{
			"valid config no constraint",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: DefaultCommandConfig(),
				Constraints:     model.Constraints{},
			},
			false,
			nil,
		},
		{
			"valid constraint no config",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: model.CommandConfig{},
				Constraints:     DefaultConstraints(),
			},
			false,
			nil,
		},
		{
			"valid constraint valid config",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: DefaultCommandConfig(),
				Constraints:     DefaultConstraints(),
			},
			false,
			nil,
		},
		{
			"global valid constraint no config",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				WorkspaceID:     nil,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: model.CommandConfig{},
				Constraints:     DefaultConstraints(),
			},
			true,
			nil,
		},
		{
			"global valid config no constraint",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: DefaultCommandConfig(),
				Constraints:     model.Constraints{},
			},
			true,
			nil,
		},
		{
			"global valid constraint valid config",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: DefaultCommandConfig(),
				Constraints:     DefaultConstraints(),
			},
			true,
			nil,
		},
		{
			"global no constraint no config",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.NTSCType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: model.CommandConfig{},
				Constraints:     model.Constraints{},
			},
			true,
			nil,
		},
		{
			"experiment workload type for NTSC policies",
			&model.NTSCTaskConfigPolicies{
				WorkloadType:    model.ExperimentType,
				LastUpdatedBy:   user.ID,
				LastUpdatedTime: time.Now().UTC().Truncate(time.Second),
				InvariantConfig: model.CommandConfig{},
				Constraints:     model.Constraints{},
			},
			true,
			ptrs.Ptr("invalid workload type"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var w model.Workspace
			if !test.global {
				w = model.Workspace{Name: uuid.NewString(), UserID: user.ID}
				_, err := db.Bun().NewInsert().Model(&w).Exec(ctx)
				require.NoError(t, err)
				workspaceIDs = append(workspaceIDs, int32(w.ID))
				test.ntscTCPs.WorkspaceID = ptrs.Ptr(w.ID)
			}

			// Test add NTSC task config policies.
			err := SetNTSCConfigPolicies(ctx, test.ntscTCPs)
			if test.err != nil {
				require.ErrorContains(t, err, *test.err)
				return
			}
			require.NoError(t, err)

			// Test get NTSC task config policies.
			ntscTCPs, err := GetNTSCConfigPolicies(ctx, test.ntscTCPs.WorkspaceID)
			require.NoError(t, err)
			ntscTCPs.LastUpdatedTime = ntscTCPs.LastUpdatedTime.UTC()
			require.Equal(t, test.ntscTCPs, ntscTCPs)

			// Test update NTSC task config policies.
			test.ntscTCPs.InvariantConfig.Environment.Image = model.RuntimeItem{
				CPU: uuid.NewString(),
			}
			err = SetNTSCConfigPolicies(ctx, test.ntscTCPs)
			require.NoError(t, err)

			// Test get NTSC task config policies.
			ntscTCPs, err = GetNTSCConfigPolicies(ctx, test.ntscTCPs.WorkspaceID)
			require.NoError(t, err)
			ntscTCPs.LastUpdatedTime = ntscTCPs.LastUpdatedTime.UTC().Truncate(time.Second)
			require.Equal(t, test.ntscTCPs, ntscTCPs)
		})
	}

	// Test invalid workspace ID.
	err := SetNTSCConfigPolicies(ctx, &model.NTSCTaskConfigPolicies{
		WorkspaceID:     ptrs.Ptr(-1),
		LastUpdatedBy:   user.ID,
		WorkloadType:    model.NTSCType,
		InvariantConfig: DefaultCommandConfig(),
		Constraints:     DefaultConstraints(),
	})
	require.ErrorContains(t, err, "violates foreign key constraint")
}

func TestWorkspaceGetPriorityLimit(t *testing.T) {
	ctx := context.Background()
	require.NoError(t, etc.SetRootPath(db.RootFromDB))
	pgDB, cleanup := db.MustResolveNewPostgresDatabase(t)
	defer cleanup()
	db.MustMigrateTestPostgres(t, pgDB, db.MigrationsFromDB)
	user := db.RequireMockUser(t, pgDB)

	// add a workspace to use
	w := model.Workspace{Name: uuid.NewString(), UserID: user.ID}
	_, err := db.Bun().NewInsert().Model(&w).Exec(ctx)
	require.NoError(t, err)
	defer func() {
		err := db.CleanupMockWorkspace([]int32{int32(w.ID)})
		if err != nil {
			log.Errorf("error when cleaning up mock workspaces")
		}
	}()

	// add priority limit for workspace NTSC
	wkspLimit := 20
	wkspInput := model.NTSCTaskConfigPolicies{
		WorkloadType: model.NTSCType,
		WorkspaceID:  &w.ID,
		Constraints: model.Constraints{
			PriorityLimit: &wkspLimit,
		},
		LastUpdatedBy: user.ID,
	}
	err = SetNTSCConfigPolicies(ctx, &wkspInput)
	require.NoError(t, err)

	// get priority limit; should match workspace
	res, found, err := GetPriorityLimit(ctx, &w.ID, model.NTSCType)
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, wkspLimit, res)

	// get limit for workspace that does not exist
	wkspIDDoesNotExist := 404
	res, found, err = GetPriorityLimit(ctx, &wkspIDDoesNotExist, model.NTSCType)
	require.NoError(t, err)
	require.False(t, found)

	// read global
	res, found, err = GetPriorityLimit(ctx, nil, model.NTSCType)
	require.NoError(t, err)
	require.False(t, found)

	// read experiment
	res, found, err = GetPriorityLimit(ctx, &w.ID, model.ExperimentType)
	require.NoError(t, err)
	require.False(t, found)

	// read invalid workload type
	res, found, err = GetPriorityLimit(ctx, &w.ID, "bogus")
	require.Error(t, err)
	require.False(t, found)
}

func TestGlobalGetPriorityLimit(t *testing.T) {
	ctx := context.Background()
	require.NoError(t, etc.SetRootPath(db.RootFromDB))
	pgDB, cleanup := db.MustResolveNewPostgresDatabase(t)
	defer cleanup()
	db.MustMigrateTestPostgres(t, pgDB, db.MigrationsFromDB)
	user := db.RequireMockUser(t, pgDB)

	// add a workspace to use
	w := model.Workspace{Name: uuid.NewString(), UserID: user.ID}
	_, err := db.Bun().NewInsert().Model(&w).Exec(ctx)
	require.NoError(t, err)
	defer func() {
		err := db.CleanupMockWorkspace([]int32{int32(w.ID)})
		if err != nil {
			log.Errorf("error when cleaning up mock workspaces")
		}
	}()

	// add priority limit for global NTSC
	globalLimit := 5
	wkspInput := model.NTSCTaskConfigPolicies{
		WorkloadType: model.NTSCType,
		WorkspaceID:  nil,
		Constraints: model.Constraints{
			PriorityLimit: &globalLimit,
		},
		LastUpdatedBy: user.ID,
	}
	err = SetNTSCConfigPolicies(ctx, &wkspInput)
	require.NoError(t, err)

	// get priority limit
	res, found, err := GetPriorityLimit(ctx, nil, model.NTSCType)
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, globalLimit, res)

	// read experiment
	res, found, err = GetPriorityLimit(ctx, nil, model.ExperimentType)
	require.NoError(t, err)
	require.False(t, found)

	// read invalid workload type
	res, found, err = GetPriorityLimit(ctx, nil, "bogus")
	require.Error(t, err)
	require.False(t, found)
}

// Test the enforcement of the primary key on the task_config_polciies table.
func TestTaskConfigPoliciesUnique(t *testing.T) {
	ctx := context.Background()
	require.NoError(t, etc.SetRootPath(db.RootFromDB))
	pgDB, cleanup := db.MustResolveNewPostgresDatabase(t)
	defer cleanup()
	db.MustMigrateTestPostgres(t, pgDB, db.MigrationsFromDB)

	user := db.RequireMockUser(t, pgDB)

	// Global scope.
	_, _, ntscTCPs := CreateMockTaskConfigPolicies(ctx, t, pgDB, user, true, true, true)
	ntscTCPs.Constraints = model.Constraints{}
	expInvariantConfig, err := json.Marshal(ntscTCPs.InvariantConfig)
	require.NoError(t, err)

	count, err := db.Bun().NewSelect().
		Table("task_config_policies").
		Where("workspace_id IS NULL").
		Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	_, err = db.Bun().NewInsert().Model(ntscTCPs).
		Where("workspace_id = ?", nil).
		Where("workspace_type = ?", model.ExperimentType).
		Value("invariant_config", "?", string(expInvariantConfig)).
		Exec(ctx)
	require.ErrorContains(t, err, "duplicate key value violates unique constraint")

	// Workspace-level.
	w, _, ntscTCPs := CreateMockTaskConfigPolicies(ctx, t, pgDB, user, false, true, true)
	ntscTCPs.Constraints = model.Constraints{}
	expInvariantConfig, err = json.Marshal(ntscTCPs.InvariantConfig)
	require.NoError(t, err)

	count, err = db.Bun().NewSelect().
		Table("task_config_policies").
		Where("workspace_id = ?", w.ID).
		Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	_, err = db.Bun().NewInsert().Model(ntscTCPs).
		Where("workspace_id = ?", w.ID).
		Where("workspace_type = ?", model.NTSCType).
		Value("invariant_config", "?", string(expInvariantConfig)).
		Exec(ctx)
	require.ErrorContains(t, err, "duplicate key value violates unique constraint")
}

func TestDeleteConfigPolicies(t *testing.T) {
	ctx := context.Background()
	require.NoError(t, etc.SetRootPath(db.RootFromDB))
	pgDB, cleanup := db.MustResolveNewPostgresDatabase(t)
	defer cleanup()
	db.MustMigrateTestPostgres(t, pgDB, db.MigrationsFromDB)

	user := db.RequireMockUser(t, pgDB)
	workspaceIDs := []int32{}

	defer func() {
		err := db.CleanupMockWorkspace(workspaceIDs)
		if err != nil {
			log.Errorf("error when cleaning up mock workspaces")
		}
	}()

	tests := []struct {
		name               string
		global             bool
		workloadType       string
		hasInvariantConfig bool
		hasConstraints     bool
		err                *string
	}{
		{"ntsc config no constraint", false, model.NTSCType, true, false, nil},
		{"ntsc config and constraint", false, model.NTSCType, true, true, nil},
		{"ntsc no config has constraint", false, model.NTSCType, false, true, nil},
		{
			"unspecified workload type", false, model.UnknownType, true, true,
			ptrs.Ptr("invalid input value for enum"),
		},
		{"ntsc no config no constraint", false, model.NTSCType, false, false, nil},
		{"global ntsc config no constraint", true, model.NTSCType, true, false, nil},
		{"global ntsc config and constraint", true, model.NTSCType, true, true, nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			w, _, ntscTCP := CreateMockTaskConfigPolicies(ctx, t, pgDB, user, test.global,
				test.hasInvariantConfig, test.hasConstraints)
			if !test.global {
				workspaceIDs = append(workspaceIDs, int32(w.ID))
			}

			err := DeleteConfigPolicies(ctx, ntscTCP.WorkspaceID, test.workloadType)
			if test.err == nil {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, *test.err)
			}
		})
	}

	// Verify that trying to delete task config policies for a nonexistent scope doesn't error out.
	err := DeleteConfigPolicies(ctx, ptrs.Ptr(-1), model.ExperimentType)
	require.NoError(t, err)

	// Verify that trying to delete task config policies for a workspace with no set policies
	// doesn't error out.
	w := &model.Workspace{Name: uuid.NewString(), UserID: user.ID}
	_, err = db.Bun().NewInsert().Model(w).Exec(ctx)
	require.NoError(t, err)
	workspaceIDs = append(workspaceIDs, int32(w.ID))

	err = DeleteConfigPolicies(ctx, ptrs.Ptr(w.ID), model.ExperimentType)
	require.NoError(t, err)

	// Verify that we can create and delete task config policies individually for different
	// workspaces.
	w1, _, _ := CreateMockTaskConfigPolicies(ctx, t, pgDB, user, false, true, true)
	workspaceIDs = append(workspaceIDs, int32(w1.ID))
	q1 := db.Bun().NewSelect().Table("task_config_policies").Where("workspace_id = ?", w1.ID)
	count, err := q1.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	w2, _, _ := CreateMockTaskConfigPolicies(ctx, t, pgDB, user, false, true, true)
	workspaceIDs = append(workspaceIDs, int32(w2.ID))
	q2 := db.Bun().NewSelect().Table("task_config_policies").Where("workspace_id = ?", w2.ID)
	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	err = DeleteConfigPolicies(ctx, ptrs.Ptr(w1.ID), model.ExperimentType)
	require.NoError(t, err)

	// Verify that exactly 1 task config policy was deleted from w1.
	count, err = q1.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	// Verify that both task config policies in w2 still exist.
	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	err = DeleteConfigPolicies(ctx, ptrs.Ptr(w2.ID), model.ExperimentType)
	require.NoError(t, err)

	// Verify that exactly 1 task config policy was deleted from w2.
	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	err = DeleteConfigPolicies(ctx, ptrs.Ptr(w1.ID), model.NTSCType)
	require.NoError(t, err)

	// Verify that no task config policies exist for w1.
	count, err = q1.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, count)

	err = DeleteConfigPolicies(ctx, ptrs.Ptr(w2.ID), model.NTSCType)
	require.NoError(t, err)

	// Verify that no task config policies exist for w2.
	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, count)

	// Test delete cascade on task config policies when a workspace is deleted.
	w1, _, _ = CreateMockTaskConfigPolicies(ctx, t, pgDB, user, false, true, true)
	workspaceIDs = append(workspaceIDs, int32(w1.ID))
	q1 = db.Bun().NewSelect().Table("task_config_policies").Where("workspace_id = ?", w1.ID)
	count, err = q1.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	w2, _, _ = CreateMockTaskConfigPolicies(ctx, t, pgDB, user, false, true, true)
	workspaceIDs = append(workspaceIDs, int32(w2.ID))
	q2 = db.Bun().NewSelect().Table("task_config_policies").Where("workspace_id = ?", w2.ID)
	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	// Verify that only the task config policies of w1 were deleted when w1 was deleted.
	_, err = db.Bun().NewDelete().Model(&model.Workspace{}).Where("id = ?", w1.ID).Exec(ctx)
	require.NoError(t, err)

	count, err = q1.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, count)

	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, count)

	// Verify that both task config policies of w2 are deleted when w2 is deleted.
	_, err = db.Bun().NewDelete().Model(&model.Workspace{}).Where("id = ?", w2.ID).Exec(ctx)
	require.NoError(t, err)

	count, err = q2.Count(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, count)
}

// CreateMockTaskConfigPolicies creates experiment and NTSC invariant configs and constraints as
// requested for the specified scope.
func CreateMockTaskConfigPolicies(ctx context.Context, t *testing.T,
	pgDB *db.PgDB, user model.User, global bool, hasInvariantConfig bool,
	hasConstraints bool) (*model.Workspace, *model.ExperimentTaskConfigPolicies,
	*model.NTSCTaskConfigPolicies,
) {
	var scope *int
	var w model.Workspace
	var ntscConfig model.CommandConfig

	var constraints model.Constraints

	if !global {
		w = model.Workspace{Name: uuid.NewString(), UserID: user.ID}
		_, err := db.Bun().NewInsert().Model(&w).Exec(ctx)
		require.NoError(t, err)
		scope = ptrs.Ptr(w.ID)
	}
	if hasInvariantConfig {
		ntscConfig = DefaultCommandConfig()
	}
	if hasConstraints {
		constraints = DefaultConstraints()
	}

	ntscTCP := &model.NTSCTaskConfigPolicies{
		WorkspaceID:     scope,
		WorkloadType:    model.NTSCType,
		LastUpdatedBy:   user.ID,
		InvariantConfig: ntscConfig,
		Constraints:     constraints,
	}
	err := SetNTSCConfigPolicies(ctx, ntscTCP)
	require.NoError(t, err)

	return &w, nil, ntscTCP
}

func DefaultCommandConfig() model.CommandConfig {
	return model.CommandConfig{
		Description: "random description",
		Resources: model.ResourcesConfig{
			Slots:    4,
			MaxSlots: ptrs.Ptr(8),
		},
	}
}

func DefaultConstraints() model.Constraints {
	return model.Constraints{
		PriorityLimit: ptrs.Ptr[int](10),
		ResourceConstraints: &model.ResourceConstraints{
			MaxSlots: ptrs.Ptr(10),
		},
	}
}
