package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/pkg/errors"
	"github.com/uptrace/bun"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SetExperimentonfigPolicies adds the experiment invariant config and constraints config policies to
// the database.
func SetExperimentConfigPolicies(ctx context.Context,
	experimentTCP *model.ExperimentTaskConfigPolicies) error {
	return Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		return SetExperimentConfigPoliciesTx(ctx, &tx, experimentTCP)
	})
}

// SetExperimentConfigPoliciesTx adds the experiment invariant config and constraints config policies to
// the database.
func SetExperimentConfigPoliciesTx(ctx context.Context, tx *bun.Tx,
	experimentTCP *model.ExperimentTaskConfigPolicies) error {

	// Validate experiment invariant config and constraints.
	expInvariantConfig, err := json.Marshal(experimentTCP.InvariantConfig)
	if err != nil {
		return errors.Wrapf(err, "error marshaling experiment invariant config %v",
			expInvariantConfig)
	}

	expConstraints, err := json.Marshal(experimentTCP.Constraints)
	if err != nil {
		return errors.Wrapf(err, "error marshaling experiment constarints %v",
			expConstraints)
	}

	_, err = Bun().NewUpdate().
		Model(experimentTCP).
		Where(`workspace_id = ? AND workload_type = ?`, experimentTCP.WorkspaceID,
			model.ExperimentType).
		Exec(ctx)
	if err != nil && err == sql.ErrNoRows {
		// Insert invariant configs and constraints.
		_, err = Bun().NewInsert().Model(experimentTCP).Exec(ctx)
		if err != nil {
			return fmt.Errorf("error inserting experiment task config policy: %w", err)
		}
		return fmt.Errorf("error updating experiment task config policy: %w", err)
	}
	return nil
}

// SetNTSCConfigPolicies adds the NTSC invariant config and constraints config policies to
// the database.
func SetNTSCConfigPolicies(ctx context.Context,
	experimentTCP *model.NTSCTaskConfigPolicies) error {
	return Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		return SetNTSCConfigPoliciesTx(ctx, &tx, experimentTCP)
	})
}

// SetNTSCConfigPoliciesTx adds the NTSC invariant config and constraints config policies to
// the database.
func SetNTSCConfigPoliciesTx(ctx context.Context, tx *bun.Tx,
	ntscTCP *model.NTSCTaskConfigPolicies) error {

	// Validate NTSC invariant config and constraints.
	expInvariantConfig, err := json.Marshal(ntscTCP.InvariantConfig)
	if err != nil {
		return errors.Wrapf(err, "error handling experiment invariant config %v",
			expInvariantConfig)
	}

	expConstraints, err := json.Marshal(ntscTCP.Constraints)
	if err != nil {
		return errors.Wrapf(err, "error handling experiment constarints %v",
			expConstraints)
	}

	_, err = Bun().NewUpdate().
		Model(ntscTCP).
		Where(`workspace_id = ? AND workload_type = ?`, ntscTCP.WorkspaceID,
			model.NTSCType).
		Exec(ctx)
	if err != nil && err == sql.ErrNoRows {
		// Insert invariant configs and constraints.
		_, err = Bun().NewInsert().Model(ntscTCP).Exec(ctx)
		if err != nil {
			return fmt.Errorf("error inserting NTSC task config policy: %w", err)
		}
		return fmt.Errorf("error updating NTSC task config policy: %w", err)
	}
	return nil
}

// GetExperimentConfigPolicies retrieves the invariant experiment config and constraints for the
// given scope (global or workspace-level).
func GetExperimentConfigPolicies(ctx context.Context,
	scope *int) (*model.ExperimentTaskConfigPolicies, error) {
	experimentTCP, _, err := getConfigPolicies(ctx, scope, model.ExperimentType)
	if err != nil {
		return nil, err
	}
	return experimentTCP, nil
}

func getConfigPolicies(ctx context.Context,
	scope *int, workloadType model.WorkloadType) (*model.ExperimentTaskConfigPolicies,
	*model.NTSCTaskConfigPolicies, error) {
	var experimentTCP model.ExperimentTaskConfigPolicies
	var ntscTCP model.NTSCTaskConfigPolicies

	switch workloadType {
	case model.ExperimentType:
		err := Bun().NewSelect().
			Model(&experimentTCP).
			Where("workspace_id = ? AND workload_type = ?", scope, workloadType.String()).
			Scan(ctx)
		if err != nil {
			if scope == nil {
				return nil, nil, fmt.Errorf("error retrieving global experiment task config "+
					"policies: %w", err)
			}
			return nil, nil, fmt.Errorf("error retrieving experiment task config policies for "+
				"workspace with ID %d: %w", *scope, err)
		}
	case model.NTSCType:
		err := Bun().NewSelect().
			Model(&ntscTCP).
			Where("workspace_id = ? AND workload_type = ?", scope, workloadType.String()).
			Scan(ctx)
		if err != nil {
			if scope == nil {
				return nil, nil, fmt.Errorf("error retrieving global NTSC task config "+
					"policies: %w", err)
			}
			return nil, nil, fmt.Errorf("error retrieving NTSC task config policies for "+
				"workspace with ID %d: %w", *scope, err)
		}
	default:
		return nil, nil, status.Errorf(codes.InvalidArgument,
			"invalid workload type for config policy: %s", workloadType.String())
	}
	return &experimentTCP, &ntscTCP, nil
}

// GetNTSCConfigPolicies retrieves the invariant NTSC config and constraints for the
// given scope (global or workspace-level).
func GetNTSCConfigPolicies(ctx context.Context,
	scope *int) (*model.NTSCTaskConfigPolicies, error) {
	_, ntscTCP, err := getConfigPolicies(ctx, scope, model.NTSCType)
	if err != nil {
		return nil, err
	}
	return ntscTCP, nil
}

// DeleteExperimentConfigPolicies deletes the invariant experiment config and constraints for the
// given scope (global or workspace-level) and workload type.
func DeleteConfigPolicies(ctx context.Context,
	scope *int, workloadType model.WorkloadType) error {
	if workloadType == model.UnknownType {
		return status.Error(codes.InvalidArgument,
			"invalid workload type for config policy: "+workloadType.String())
	}
	_, err := Bun().NewDelete().
		Table("task_config_policies").
		Where("workspace_id = ? AND workload_type = ?", scope, workloadType.String()).
		Exec(ctx)
	if err != nil {
		if scope == nil {
			return fmt.Errorf("error deleting global %s config policies:%w",
				strings.ToLower(workloadType.String()), err)
		}
		return fmt.Errorf("error deleting %s config policies for workspace with ID %d: %w",
			strings.ToLower(workloadType.String()), *scope, err)
	}
	return nil
}
