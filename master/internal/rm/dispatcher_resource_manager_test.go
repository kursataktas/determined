package rm

import (
	"testing"
	"time"

	launcher "github.hpe.com/hpe/hpc-ard-launcher-go/launcher"
	"gotest.tools/assert"

	"github.com/determined-ai/determined/master/internal/config"
	"github.com/determined-ai/determined/master/pkg/actor"
	"github.com/determined-ai/determined/master/pkg/device"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/proto/pkg/agentv1"
	"github.com/determined-ai/determined/proto/pkg/containerv1"
	"github.com/determined-ai/determined/proto/pkg/devicev1"
	"github.com/determined-ai/determined/proto/pkg/resourcepoolv1"
)

func Test_authContext(t *testing.T) {
	m := &dispatcherResourceManager{
		authToken: "xyz",
	}
	ctx := &actor.Context{}
	ctxWith := m.authContext(ctx)
	if authToken := ctxWith.Value(launcher.ContextAccessToken); authToken != nil {
		assert.Equal(t, authToken, "xyz")
		return
	}
	t.Errorf("authContext failed")
}

func Test_generateGetAgentsResponse(t *testing.T) {
	n1 := hpcNodeDetails{
		Partitions:    []string{"Partition 1"},
		Addresses:     []string{"address 1", "address 2"},
		Draining:      true,
		Allocated:     true,
		Name:          "Node 1",
		GpuCount:      0,
		GpuInUseCount: 0,
		CPUCount:      8,
		CPUInUseCount: 6,
	}

	n2 := hpcNodeDetails{
		Partitions:    []string{"Partition 1", "Partition 2"},
		Addresses:     []string{"address 3", "address 4"},
		Draining:      false,
		Allocated:     true,
		Name:          "Node 2",
		GpuCount:      2,
		GpuInUseCount: 1,
		CPUCount:      8,
		CPUInUseCount: 0,
	}

	n3 := hpcNodeDetails{
		Partitions:    []string{"Partition 1", "Partition 3"},
		Addresses:     []string{"address 3", "address 4"},
		Draining:      false,
		Allocated:     true,
		Name:          "Node 3",
		GpuCount:      2,
		GpuInUseCount: 1,
		CPUCount:      8,
		CPUInUseCount: 0,
	}

	nodes := []hpcNodeDetails{n1, n2, n3}

	hpcResource := &hpcResources{
		Nodes: nodes,
	}

	rocm := device.ROCM
	cuda := device.CUDA
	overrides := map[string]config.DispatcherPartitionOverrideConfigs{
		"Partition 2": {
			SlotType: &rocm,
		},
		"Partition 3": {
			SlotType: &cuda,
		},
	}

	config := &config.DispatcherResourceManagerConfig{
		PartitionOverrides: overrides,
	}

	m := &dispatcherResourceManager{
		config: config,
		resourceDetails: hpcResourceDetailsCache{
			lastSample: *hpcResource,
			sampleTime: time.Now(),
		},
	}

	want0 := map[string]*agentv1.Slot{
		"/agents/Node 1/slots/0": {
			Id:        "0",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/1": {
			Id:        "1",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/2": {
			Id:        "2",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/3": {
			Id:        "3",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/4": {
			Id:        "4",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/5": {
			Id:        "5",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},
		"/agents/Node 1/slots/6": {
			Id:      "6",
			Device:  &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled: true,
		},
		"/agents/Node 1/slots/7": {
			Id:      "7",
			Device:  &devicev1.Device{Type: devicev1.Type_TYPE_CPU},
			Enabled: true,
		},
	}

	want1 := map[string]*agentv1.Slot{
		"/agents/Node 2/slots/0": {
			Id:        "0",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_ROCM},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},

		"/agents/Node 2/slots/1": {
			Id:      "1",
			Device:  &devicev1.Device{Type: devicev1.Type_TYPE_ROCM},
			Enabled: true,
		},
	}

	want2 := map[string]*agentv1.Slot{
		"/agents/Node 3/slots/0": {
			Id:        "0",
			Device:    &devicev1.Device{Type: devicev1.Type_TYPE_CUDA},
			Enabled:   true,
			Container: &containerv1.Container{State: containerv1.State_STATE_RUNNING},
		},

		"/agents/Node 3/slots/1": {
			Id:      "1",
			Device:  &devicev1.Device{Type: devicev1.Type_TYPE_CUDA},
			Enabled: true,
		},
	}

	wantSlots := []map[string]*agentv1.Slot{want0, want1, want2}

	ctx := &actor.Context{}
	resp := m.generateGetAgentsResponse(ctx)
	assert.Equal(t, len(resp.Agents), len(nodes))

	for i, agent := range resp.Agents {
		assert.Equal(t, agent.Id, nodes[i].Name)
		assert.DeepEqual(t, agent.ResourcePools, nodes[i].Partitions)
		assert.DeepEqual(t, agent.Addresses, nodes[i].Addresses)
		assert.Equal(t, agent.Draining, nodes[i].Draining)

		assert.Equal(t, len(agent.Slots), len(wantSlots[i]))
		for key, value := range agent.Slots {
			wantValue := wantSlots[i][key]
			assert.Equal(t, value.Id, wantValue.Id)
			assert.Equal(t, value.Device.Type, wantValue.Device.Type)
			assert.Equal(t, value.Enabled, wantValue.Enabled)

			if wantValue.Container != nil {
				assert.Equal(t, value.Container.State, wantValue.Container.State)
			} else if value.Container != nil {
				t.Errorf("agent.Slots %s Container value error", key)
			}
		}
	}
}

func Test_summarizeResourcePool(t *testing.T) {
	type args struct {
		ctx     *actor.Context
		wlmType string
	}

	type want struct {
		pools         []resourcepoolv1.ResourcePool
		location      string
		schedulerType resourcepoolv1.SchedulerType
		fittingPolicy resourcepoolv1.FittingPolicy
	}

	p1 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "partition 1",
		IsDefault:              true,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 2,
		TotalNodes:             10,
		TotalGpuSlots:          5,
	}
	p2 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "partition 2",
		IsDefault:              false,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 0,
		TotalNodes:             12,
		TotalGpuSlots:          0,
		TotalCPUSlots:          20,
		TotalAvailableCPUSlots: 8,
	}
	p3 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "partition 3",
		IsDefault:              false,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 0,
		TotalNodes:             15,
		TotalGpuSlots:          7,
	}

	tests := []struct {
		name       string
		partitions []hpcPartitionDetails
		args       args
		want       want
	}{
		{
			name:       "One resource pool test",
			partitions: []hpcPartitionDetails{p1},
			args: args{
				ctx:     &actor.Context{},
				wlmType: slurmSchedulerType,
			},
			want: want{
				pools: []resourcepoolv1.ResourcePool{
					{
						Name:           "partition 1",
						SlotType:       devicev1.Type_TYPE_CUDA,
						SlotsAvailable: 5,
						SlotsUsed:      3,
						NumAgents:      10,
					},
				},
				location:      "Slurm",
				schedulerType: resourcepoolv1.SchedulerType_SCHEDULER_TYPE_SLURM,
				fittingPolicy: resourcepoolv1.FittingPolicy_FITTING_POLICY_SLURM,
			},
		},
		{
			name:       "Two resource pool test; WLM is PBS",
			partitions: []hpcPartitionDetails{p1, p2},
			args: args{
				ctx:     &actor.Context{},
				wlmType: pbsSchedulerType,
			},
			want: want{
				pools: []resourcepoolv1.ResourcePool{
					{
						Name:           "partition 1",
						SlotType:       devicev1.Type_TYPE_CUDA,
						SlotsAvailable: 5,
						SlotsUsed:      3,
						NumAgents:      10,
					},
					{
						Name:           "partition 2",
						SlotType:       devicev1.Type_TYPE_CPU,
						SlotsAvailable: 20,
						SlotsUsed:      12,
						NumAgents:      12,
						SlotsPerAgent:  1,
					},
				},
				location:      "PBS",
				schedulerType: resourcepoolv1.SchedulerType_SCHEDULER_TYPE_PBS,
				fittingPolicy: resourcepoolv1.FittingPolicy_FITTING_POLICY_PBS,
			},
		},
		{
			name:       "Three resource pool test",
			partitions: []hpcPartitionDetails{p1, p2, p3},
			args: args{
				ctx:     &actor.Context{},
				wlmType: "mystery",
			},
			want: want{
				pools: []resourcepoolv1.ResourcePool{
					{
						Name:           "partition 1",
						SlotType:       devicev1.Type_TYPE_CUDA,
						SlotsAvailable: 5,
						SlotsUsed:      3,
						NumAgents:      10,
					},
					{
						Name:           "partition 2",
						SlotType:       devicev1.Type_TYPE_CPU,
						SlotsAvailable: 20,
						SlotsUsed:      12,
						NumAgents:      12,
						SlotsPerAgent:  1,
					},
					{
						Name:           "partition 3",
						SlotType:       devicev1.Type_TYPE_CUDA,
						SlotsAvailable: 7,
						SlotsUsed:      7,
						NumAgents:      15,
					},
				},
				location:      "Unknown",
				schedulerType: resourcepoolv1.SchedulerType_SCHEDULER_TYPE_UNSPECIFIED,
				fittingPolicy: resourcepoolv1.FittingPolicy_FITTING_POLICY_UNSPECIFIED,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hpcResource := &hpcResources{
				Partitions: tt.partitions,
			}

			overrides := make(map[string]config.DispatcherPartitionOverrideConfigs)
			config := &config.DispatcherResourceManagerConfig{
				PartitionOverrides: overrides,
			}

			m := &dispatcherResourceManager{
				config: config,
				resourceDetails: hpcResourceDetailsCache{
					lastSample: *hpcResource,
					sampleTime: time.Now(),
				},
				wlmType: tt.args.wlmType,
			}

			res, _ := m.summarizeResourcePool(tt.args.ctx)

			assert.Equal(t, len(tt.want.pools), len(res))
			for i, pool := range res {
				assert.Equal(t, pool.Name, tt.want.pools[i].Name)
				assert.Equal(t, pool.SlotType, tt.want.pools[i].SlotType)
				assert.Equal(t, pool.SlotsAvailable, tt.want.pools[i].SlotsAvailable)
				assert.Equal(t, pool.SlotsUsed, tt.want.pools[i].SlotsUsed)
				assert.Equal(t, pool.NumAgents, tt.want.pools[i].NumAgents)

				assert.Equal(t, pool.Description, tt.want.location+"-managed pool of resources")
				assert.Equal(t, pool.Type, resourcepoolv1.ResourcePoolType_RESOURCE_POOL_TYPE_STATIC)
				assert.Equal(t, pool.SlotsPerAgent, tt.want.pools[i].SlotsPerAgent)
				assert.Equal(t, pool.AuxContainerCapacityPerAgent, int32(0))
				assert.Equal(t, pool.SchedulerType, tt.want.schedulerType)
				assert.Equal(t, pool.SchedulerFittingPolicy, tt.want.fittingPolicy)
				assert.Equal(t, pool.Location, tt.want.location)
				assert.Equal(t, pool.InstanceType, tt.want.location)
				assert.Equal(t, pool.ImageId, "")
			}
		})
	}
}

func Test_dispatcherResourceManager_selectDefaultPools(t *testing.T) {
	type fields struct {
		config                   *config.DispatcherResourceManagerConfig
		apiClient                *launcher.APIClient
		hpcResourcesManifest     *launcher.Manifest
		reqList                  *taskList
		groups                   map[*actor.Ref]*group
		slotsUsedPerGroup        map[*group]int
		dispatchIDToAllocationID map[string]model.AllocationID
		masterTLSConfig          model.TLSClientConfig
		loggingConfig            model.LoggingConfig
		jobWatcher               *launcherMonitor
		authToken                string
	}
	type args struct {
		ctx                *actor.Context
		hpcResourceDetails []hpcPartitionDetails
	}

	p1 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "worf",
		IsDefault:              true,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 0,
		TotalNodes:             0,
		TotalGpuSlots:          0,
	}
	p2 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "data",
		IsDefault:              false,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 0,
		TotalNodes:             0,
		TotalGpuSlots:          1,
	}
	p3 := hpcPartitionDetails{
		TotalAvailableNodes:    0,
		PartitionName:          "picard",
		IsDefault:              false,
		TotalAllocatedNodes:    0,
		TotalAvailableGpuSlots: 0,
		TotalNodes:             0,
		TotalGpuSlots:          0,
	}
	hpc := []hpcPartitionDetails{
		p1,
	}
	hpc2 := []hpcPartitionDetails{
		p1, p2,
	}
	hpc3 := []hpcPartitionDetails{
		p1, p2, p3,
	}
	// One partition, no GPUs
	hpc4 := []hpcPartitionDetails{
		p3,
	}

	tests := []struct {
		name        string
		fields      fields
		args        args
		wantCompute string
		wantAux     string
	}{
		{
			name:        "One partition test",
			fields:      fields{},
			args:        args{hpcResourceDetails: hpc},
			wantCompute: "worf",
			wantAux:     "worf",
		},
		{
			name:        "Two partition test",
			fields:      fields{},
			args:        args{hpcResourceDetails: hpc2},
			wantCompute: "data",
			wantAux:     "worf",
		},
		{
			name:        "Three partition test",
			fields:      fields{},
			args:        args{hpcResourceDetails: hpc3},
			wantCompute: "data",
			wantAux:     "worf",
		},
		{
			name:        "No GPU partition test",
			fields:      fields{},
			args:        args{hpcResourceDetails: hpc4},
			wantCompute: "picard",
			wantAux:     "picard",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &dispatcherResourceManager{
				config:                   tt.fields.config,
				apiClient:                tt.fields.apiClient,
				hpcResourcesManifest:     tt.fields.hpcResourcesManifest,
				reqList:                  tt.fields.reqList,
				groups:                   tt.fields.groups,
				slotsUsedPerGroup:        tt.fields.slotsUsedPerGroup,
				dispatchIDToAllocationID: tt.fields.dispatchIDToAllocationID,
				masterTLSConfig:          tt.fields.masterTLSConfig,
				loggingConfig:            tt.fields.loggingConfig,
				jobWatcher:               tt.fields.jobWatcher,
				authToken:                tt.fields.authToken,
			}
			compute, aux := m.selectDefaultPools(tt.args.ctx, tt.args.hpcResourceDetails)
			if compute != tt.wantCompute {
				t.Errorf("selectDefaultPools() compute got = %v, want %v", compute, tt.wantCompute)
			}
			if aux != tt.wantAux {
				t.Errorf("selectDefaultPools() aux got = %v, want %v", aux, tt.wantAux)
			}
		})
	}
}

func Test_dispatcherResourceManager_determineWlmType(t *testing.T) {
	type fields struct{}
	type args struct {
		dispatchInfo  launcher.DispatchInfo
		ctx           *actor.Context
		reporter      string
		message       string
		wantedWlmType string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name:   "Expect Slurm",
			fields: fields{},
			args: args{
				ctx:           &actor.Context{},
				reporter:      slurmResourcesCarrier,
				message:       "Successfully launched payload x",
				wantedWlmType: "slurm",
			},
		},
		{
			name:   "Expect PBS",
			fields: fields{},
			args: args{
				ctx:           &actor.Context{},
				reporter:      pbsResourcesCarrier,
				message:       "Successfully launched payload x",
				wantedWlmType: "pbs",
			},
		},
		{
			name:   "PBS ran, but failed",
			fields: fields{},
			args: args{
				ctx:           &actor.Context{},
				reporter:      pbsResourcesCarrier,
				message:       "UnSuccessfully launched payload x",
				wantedWlmType: "",
			},
		},
		{
			name:   "Neither PBS nor Slurm responded",
			fields: fields{},
			args: args{
				ctx:           &actor.Context{},
				reporter:      "com.cray.analytics.capsules.carriers.hpc.other.OtherResources",
				message:       "Successfully launched payload x",
				wantedWlmType: "",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &dispatcherResourceManager{}
			someReporter := "someone"
			someMessage := "the message"
			event1 := launcher.Event{
				Reporter: &someReporter,
				Message:  &someMessage,
			}
			event2 := launcher.Event{
				Reporter: &tt.args.reporter,
				Message:  &tt.args.message,
			}

			di := launcher.DispatchInfo{
				Events: &[]launcher.Event{event1, event2},
			}

			m.determineWlmType(di, tt.args.ctx)

			if m.wlmType != tt.args.wantedWlmType {
				t.Errorf("selectDefaultPools() compute got = %v, want %v", m.wlmType, tt.args.wantedWlmType)
			}
		})
	}
}

func Test_dispatcherResourceManager_checkLauncherVersion(t *testing.T) {
	assert.Equal(t, checkMinimumLauncherVersion("4.1.0"), true)
	assert.Equal(t, checkMinimumLauncherVersion("3.2.0"), true)
	assert.Equal(t, checkMinimumLauncherVersion("3.1.3"), true)
	assert.Equal(t, checkMinimumLauncherVersion("3.1.3-SNAPSHOT"), true)
	assert.Equal(t, checkMinimumLauncherVersion("3.1.0"), false)
	assert.Equal(t, checkMinimumLauncherVersion("2.3.3"), false)
	assert.Equal(t, checkMinimumLauncherVersion("3.0.3"), false)
	assert.Equal(t, checkMinimumLauncherVersion("x.y.z"), false)
	assert.Equal(t, checkMinimumLauncherVersion("abc"), false)
}
