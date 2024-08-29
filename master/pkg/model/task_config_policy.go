package model

import (
	"time"

	"github.com/uptrace/bun"
)

type WorkloadType string

// Constants.

const (
	// UnknownType constant.
	UnknownType WorkloadType = "UNKNOWN"
	// ExperimentType constant.
	ExperimentType WorkloadType = "EXPERIMENT"
	// NTSCType constant.
	NTSCType WorkloadType = "NTSC"
)

// InvariantConfigConstraint is the bun model of a task config policy.
type InvariantConfigConstraint struct {
	bun.BaseModel   `bun:"table:invariant_config_constraints"`
	WorkspaceID     int          `bun:"workspace_id,unique"`
	LastUpdatedBy   int          `bun:"last_updated_by,notnull"`
	LastUpdatedTime time.Time    `bun:"last_updated_time,notnull"`
	WorkloadType    WorkloadType `bun:"workload_type,notnull"`
	InvariantConfig *string      `bun:"invariant_config"`
	Constraints     *string      `bun:"constraints"`
}
