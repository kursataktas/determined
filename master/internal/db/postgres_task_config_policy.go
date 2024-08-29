package db

import (
	"context"

	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/uptrace/bun"
)

func AddTaskConfigPolicy(ctx context.Context,
	taskConfigPolicy *model.InvariantConfigConstraint) error {
	return Bun().RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		return AddTaskConfigPolicyTx(ctx, &tx, taskConfigPolicy)
	})
}

func AddTaskConfigPolicyTx(ctx context.Context, tx *bun.Tx,
	taskConfigPolicy *model.InvariantConfigConstraint) error {
	_, err := Bun().NewInsert().Model(taskConfigPolicy).Exec(ctx)
	return err
}
