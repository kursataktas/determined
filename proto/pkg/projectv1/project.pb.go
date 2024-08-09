// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// source: determined/project/v1/project.proto

package projectv1

import (
	workspacev1 "github.com/determined-ai/determined/proto/pkg/workspacev1"
	timestamp "github.com/golang/protobuf/ptypes/timestamp"
	wrappers "github.com/golang/protobuf/ptypes/wrappers"
	_ "github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger/options"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// LocationType indicates where a column comes from
type LocationType int32

const (
	// Location unknown
	LocationType_LOCATION_TYPE_UNSPECIFIED LocationType = 0
	// Column is located on the experiment
	LocationType_LOCATION_TYPE_EXPERIMENT LocationType = 1
	// Column is located in the hyperparameter config of the experiment
	LocationType_LOCATION_TYPE_HYPERPARAMETERS LocationType = 2
	// Column is located on the experiment's validation metrics
	LocationType_LOCATION_TYPE_VALIDATIONS LocationType = 3
	// Column is located on the experiment's training steps
	LocationType_LOCATION_TYPE_TRAINING LocationType = 4
	// Column is located on the experiment's custom metric
	LocationType_LOCATION_TYPE_CUSTOM_METRIC LocationType = 5
	// Column is located on the run
	LocationType_LOCATION_TYPE_RUN LocationType = 6
	// Column is located in the hyperparameter of the run
	LocationType_LOCATION_TYPE_RUN_HYPERPARAMETERS LocationType = 7
	// Column is located on the run's arbitrary metadata
	LocationType_LOCATION_TYPE_RUN_METADATA LocationType = 8
)

// Enum value maps for LocationType.
var (
	LocationType_name = map[int32]string{
		0: "LOCATION_TYPE_UNSPECIFIED",
		1: "LOCATION_TYPE_EXPERIMENT",
		2: "LOCATION_TYPE_HYPERPARAMETERS",
		3: "LOCATION_TYPE_VALIDATIONS",
		4: "LOCATION_TYPE_TRAINING",
		5: "LOCATION_TYPE_CUSTOM_METRIC",
		6: "LOCATION_TYPE_RUN",
		7: "LOCATION_TYPE_RUN_HYPERPARAMETERS",
		8: "LOCATION_TYPE_RUN_METADATA",
	}
	LocationType_value = map[string]int32{
		"LOCATION_TYPE_UNSPECIFIED":         0,
		"LOCATION_TYPE_EXPERIMENT":          1,
		"LOCATION_TYPE_HYPERPARAMETERS":     2,
		"LOCATION_TYPE_VALIDATIONS":         3,
		"LOCATION_TYPE_TRAINING":            4,
		"LOCATION_TYPE_CUSTOM_METRIC":       5,
		"LOCATION_TYPE_RUN":                 6,
		"LOCATION_TYPE_RUN_HYPERPARAMETERS": 7,
		"LOCATION_TYPE_RUN_METADATA":        8,
	}
)

func (x LocationType) Enum() *LocationType {
	p := new(LocationType)
	*p = x
	return p
}

func (x LocationType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LocationType) Descriptor() protoreflect.EnumDescriptor {
	return file_determined_project_v1_project_proto_enumTypes[0].Descriptor()
}

func (LocationType) Type() protoreflect.EnumType {
	return &file_determined_project_v1_project_proto_enumTypes[0]
}

func (x LocationType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LocationType.Descriptor instead.
func (LocationType) EnumDescriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{0}
}

// ColumnType indicates the type of data under the column
type ColumnType int32

const (
	// data type is unknown/mixed
	ColumnType_COLUMN_TYPE_UNSPECIFIED ColumnType = 0
	// data type is textual
	ColumnType_COLUMN_TYPE_TEXT ColumnType = 1
	// data type is numeric
	ColumnType_COLUMN_TYPE_NUMBER ColumnType = 2
	// data type is a date
	ColumnType_COLUMN_TYPE_DATE ColumnType = 3
	// data type is an array
	ColumnType_COLUMN_TYPE_ARRAY ColumnType = 4
)

// Enum value maps for ColumnType.
var (
	ColumnType_name = map[int32]string{
		0: "COLUMN_TYPE_UNSPECIFIED",
		1: "COLUMN_TYPE_TEXT",
		2: "COLUMN_TYPE_NUMBER",
		3: "COLUMN_TYPE_DATE",
		4: "COLUMN_TYPE_ARRAY",
	}
	ColumnType_value = map[string]int32{
		"COLUMN_TYPE_UNSPECIFIED": 0,
		"COLUMN_TYPE_TEXT":        1,
		"COLUMN_TYPE_NUMBER":      2,
		"COLUMN_TYPE_DATE":        3,
		"COLUMN_TYPE_ARRAY":       4,
	}
)

func (x ColumnType) Enum() *ColumnType {
	p := new(ColumnType)
	*p = x
	return p
}

func (x ColumnType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ColumnType) Descriptor() protoreflect.EnumDescriptor {
	return file_determined_project_v1_project_proto_enumTypes[1].Descriptor()
}

func (ColumnType) Type() protoreflect.EnumType {
	return &file_determined_project_v1_project_proto_enumTypes[1]
}

func (x ColumnType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ColumnType.Descriptor instead.
func (ColumnType) EnumDescriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{1}
}

// Project Column is a description of a column used on experiments in the
// project.
type ProjectColumn struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Raw column name.
	Column string `protobuf:"bytes,1,opt,name=column,proto3" json:"column,omitempty"`
	// Where the column comes from.
	Location LocationType `protobuf:"varint,2,opt,name=location,proto3,enum=determined.project.v1.LocationType" json:"location,omitempty"`
	// Type of data in the column.
	Type ColumnType `protobuf:"varint,3,opt,name=type,proto3,enum=determined.project.v1.ColumnType" json:"type,omitempty"`
	// Human-friendly name.
	DisplayName string `protobuf:"bytes,4,opt,name=display_name,json=displayName,proto3" json:"display_name,omitempty"`
}

func (x *ProjectColumn) Reset() {
	*x = ProjectColumn{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_project_v1_project_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProjectColumn) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProjectColumn) ProtoMessage() {}

func (x *ProjectColumn) ProtoReflect() protoreflect.Message {
	mi := &file_determined_project_v1_project_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProjectColumn.ProtoReflect.Descriptor instead.
func (*ProjectColumn) Descriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{0}
}

func (x *ProjectColumn) GetColumn() string {
	if x != nil {
		return x.Column
	}
	return ""
}

func (x *ProjectColumn) GetLocation() LocationType {
	if x != nil {
		return x.Location
	}
	return LocationType_LOCATION_TYPE_UNSPECIFIED
}

func (x *ProjectColumn) GetType() ColumnType {
	if x != nil {
		return x.Type
	}
	return ColumnType_COLUMN_TYPE_UNSPECIFIED
}

func (x *ProjectColumn) GetDisplayName() string {
	if x != nil {
		return x.DisplayName
	}
	return ""
}

// Note is a user comment connected to a project.
type Note struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The name or title of the note.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The text contents of the note.
	Contents string `protobuf:"bytes,2,opt,name=contents,proto3" json:"contents,omitempty"`
}

func (x *Note) Reset() {
	*x = Note{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_project_v1_project_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Note) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Note) ProtoMessage() {}

func (x *Note) ProtoReflect() protoreflect.Message {
	mi := &file_determined_project_v1_project_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Note.ProtoReflect.Descriptor instead.
func (*Note) Descriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{1}
}

func (x *Note) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Note) GetContents() string {
	if x != nil {
		return x.Contents
	}
	return ""
}

// Project is a named collection of experiments.
type Project struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The unique id of the project.
	Id int32 `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	// The unique name of the project.
	Name string `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	// The id of the associated workspace.
	WorkspaceId int32 `protobuf:"varint,3,opt,name=workspace_id,json=workspaceId,proto3" json:"workspace_id,omitempty"`
	// The description of the project.
	Description string `protobuf:"bytes,4,opt,name=description,proto3" json:"description,omitempty"`
	// Time of most recently started experiment within this project.
	LastExperimentStartedAt *timestamp.Timestamp `protobuf:"bytes,5,opt,name=last_experiment_started_at,json=lastExperimentStartedAt,proto3" json:"last_experiment_started_at,omitempty"`
	// Notes associated with this project.
	Notes []*Note `protobuf:"bytes,6,rep,name=notes,proto3" json:"notes,omitempty"`
	// Count of experiments associated with this project.
	NumExperiments int32 `protobuf:"varint,7,opt,name=num_experiments,json=numExperiments,proto3" json:"num_experiments,omitempty"`
	// Count of active experiments associated with this project.
	NumActiveExperiments int32 `protobuf:"varint,8,opt,name=num_active_experiments,json=numActiveExperiments,proto3" json:"num_active_experiments,omitempty"`
	// Whether this project is archived or not.
	Archived bool `protobuf:"varint,9,opt,name=archived,proto3" json:"archived,omitempty"`
	// User who created this project.
	Username string `protobuf:"bytes,10,opt,name=username,proto3" json:"username,omitempty"`
	// Whether this project is immutable (default uncategorized project).
	Immutable bool `protobuf:"varint,11,opt,name=immutable,proto3" json:"immutable,omitempty"`
	// ID of the user who created this project.
	UserId int32 `protobuf:"varint,12,opt,name=user_id,json=userId,proto3" json:"user_id,omitempty"`
	// The name of the associated workspace.
	WorkspaceName string `protobuf:"bytes,13,opt,name=workspace_name,json=workspaceName,proto3" json:"workspace_name,omitempty"`
	// State of project during deletion.
	State workspacev1.WorkspaceState `protobuf:"varint,14,opt,name=state,proto3,enum=determined.workspace.v1.WorkspaceState" json:"state,omitempty"`
	// Message stored from errors on async-deleting a project.
	ErrorMessage string `protobuf:"bytes,15,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
	// The key of the project.
	Key string `protobuf:"bytes,16,opt,name=key,proto3" json:"key,omitempty"`
	// Count of runs associated with this project.
	NumRuns int32 `protobuf:"varint,17,opt,name=num_runs,json=numRuns,proto3" json:"num_runs,omitempty"`
}

func (x *Project) Reset() {
	*x = Project{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_project_v1_project_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Project) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Project) ProtoMessage() {}

func (x *Project) ProtoReflect() protoreflect.Message {
	mi := &file_determined_project_v1_project_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Project.ProtoReflect.Descriptor instead.
func (*Project) Descriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{2}
}

func (x *Project) GetId() int32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *Project) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Project) GetWorkspaceId() int32 {
	if x != nil {
		return x.WorkspaceId
	}
	return 0
}

func (x *Project) GetDescription() string {
	if x != nil {
		return x.Description
	}
	return ""
}

func (x *Project) GetLastExperimentStartedAt() *timestamp.Timestamp {
	if x != nil {
		return x.LastExperimentStartedAt
	}
	return nil
}

func (x *Project) GetNotes() []*Note {
	if x != nil {
		return x.Notes
	}
	return nil
}

func (x *Project) GetNumExperiments() int32 {
	if x != nil {
		return x.NumExperiments
	}
	return 0
}

func (x *Project) GetNumActiveExperiments() int32 {
	if x != nil {
		return x.NumActiveExperiments
	}
	return 0
}

func (x *Project) GetArchived() bool {
	if x != nil {
		return x.Archived
	}
	return false
}

func (x *Project) GetUsername() string {
	if x != nil {
		return x.Username
	}
	return ""
}

func (x *Project) GetImmutable() bool {
	if x != nil {
		return x.Immutable
	}
	return false
}

func (x *Project) GetUserId() int32 {
	if x != nil {
		return x.UserId
	}
	return 0
}

func (x *Project) GetWorkspaceName() string {
	if x != nil {
		return x.WorkspaceName
	}
	return ""
}

func (x *Project) GetState() workspacev1.WorkspaceState {
	if x != nil {
		return x.State
	}
	return workspacev1.WorkspaceState_WORKSPACE_STATE_UNSPECIFIED
}

func (x *Project) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

func (x *Project) GetKey() string {
	if x != nil {
		return x.Key
	}
	return ""
}

func (x *Project) GetNumRuns() int32 {
	if x != nil {
		return x.NumRuns
	}
	return 0
}

// PatchProject is a partial update to a project with all optional fields.
type PatchProject struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The new name for the project.
	Name *wrappers.StringValue `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The new description for the project.
	Description *wrappers.StringValue `protobuf:"bytes,2,opt,name=description,proto3" json:"description,omitempty"`
	// The new key for the project.
	Key *wrappers.StringValue `protobuf:"bytes,3,opt,name=key,proto3" json:"key,omitempty"`
}

func (x *PatchProject) Reset() {
	*x = PatchProject{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_project_v1_project_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PatchProject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PatchProject) ProtoMessage() {}

func (x *PatchProject) ProtoReflect() protoreflect.Message {
	mi := &file_determined_project_v1_project_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PatchProject.ProtoReflect.Descriptor instead.
func (*PatchProject) Descriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{3}
}

func (x *PatchProject) GetName() *wrappers.StringValue {
	if x != nil {
		return x.Name
	}
	return nil
}

func (x *PatchProject) GetDescription() *wrappers.StringValue {
	if x != nil {
		return x.Description
	}
	return nil
}

func (x *PatchProject) GetKey() *wrappers.StringValue {
	if x != nil {
		return x.Key
	}
	return nil
}

// MetricsRange represents the range of a metrics. Range is a in the format of
// [min, max].
type MetricsRange struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The name of metrics formatted as <type>.<name>.
	MetricsName string `protobuf:"bytes,1,opt,name=metrics_name,json=metricsName,proto3" json:"metrics_name,omitempty"`
	// The min of metrics values.
	Min float64 `protobuf:"fixed64,2,opt,name=min,proto3" json:"min,omitempty"`
	// The max of metrics values.
	Max float64 `protobuf:"fixed64,3,opt,name=max,proto3" json:"max,omitempty"`
}

func (x *MetricsRange) Reset() {
	*x = MetricsRange{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_project_v1_project_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *MetricsRange) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MetricsRange) ProtoMessage() {}

func (x *MetricsRange) ProtoReflect() protoreflect.Message {
	mi := &file_determined_project_v1_project_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MetricsRange.ProtoReflect.Descriptor instead.
func (*MetricsRange) Descriptor() ([]byte, []int) {
	return file_determined_project_v1_project_proto_rawDescGZIP(), []int{4}
}

func (x *MetricsRange) GetMetricsName() string {
	if x != nil {
		return x.MetricsName
	}
	return ""
}

func (x *MetricsRange) GetMin() float64 {
	if x != nil {
		return x.Min
	}
	return 0
}

func (x *MetricsRange) GetMax() float64 {
	if x != nil {
		return x.Max
	}
	return 0
}

var File_determined_project_v1_project_proto protoreflect.FileDescriptor

var file_determined_project_v1_project_proto_rawDesc = []byte{
	0x0a, 0x23, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x70, 0x72, 0x6f,
	0x6a, 0x65, 0x63, 0x74, 0x2f, 0x76, 0x31, 0x2f, 0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x15, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65,
	0x64, 0x2e, 0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x2e, 0x76, 0x31, 0x1a, 0x27, 0x64, 0x65,
	0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61,
	0x63, 0x65, 0x2f, 0x76, 0x31, 0x2f, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1f, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x77, 0x72, 0x61, 0x70, 0x70, 0x65, 0x72, 0x73,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x2c, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x2d, 0x67,
	0x65, 0x6e, 0x2d, 0x73, 0x77, 0x61, 0x67, 0x67, 0x65, 0x72, 0x2f, 0x6f, 0x70, 0x74, 0x69, 0x6f,
	0x6e, 0x73, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x22, 0xe4, 0x01, 0x0a, 0x0d, 0x50, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74,
	0x43, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x12, 0x16, 0x0a, 0x06, 0x63, 0x6f, 0x6c, 0x75, 0x6d, 0x6e,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x63, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x12, 0x3f,
	0x0a, 0x08, 0x6c, 0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0e,
	0x32, 0x23, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x70, 0x72,
	0x6f, 0x6a, 0x65, 0x63, 0x74, 0x2e, 0x76, 0x31, 0x2e, 0x4c, 0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x54, 0x79, 0x70, 0x65, 0x52, 0x08, 0x6c, 0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x12,
	0x35, 0x0a, 0x04, 0x74, 0x79, 0x70, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x21, 0x2e,
	0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x70, 0x72, 0x6f, 0x6a, 0x65,
	0x63, 0x74, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x54, 0x79, 0x70, 0x65,
	0x52, 0x04, 0x74, 0x79, 0x70, 0x65, 0x12, 0x21, 0x0a, 0x0c, 0x64, 0x69, 0x73, 0x70, 0x6c, 0x61,
	0x79, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0b, 0x64, 0x69,
	0x73, 0x70, 0x6c, 0x61, 0x79, 0x4e, 0x61, 0x6d, 0x65, 0x3a, 0x20, 0x92, 0x41, 0x1d, 0x0a, 0x1b,
	0xd2, 0x01, 0x06, 0x63, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0xd2, 0x01, 0x08, 0x6c, 0x6f, 0x63, 0x61,
	0x74, 0x69, 0x6f, 0x6e, 0xd2, 0x01, 0x04, 0x74, 0x79, 0x70, 0x65, 0x22, 0x4f, 0x0a, 0x04, 0x4e,
	0x6f, 0x74, 0x65, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x1a, 0x0a, 0x08, 0x63, 0x6f, 0x6e, 0x74, 0x65,
	0x6e, 0x74, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x63, 0x6f, 0x6e, 0x74, 0x65,
	0x6e, 0x74, 0x73, 0x3a, 0x17, 0x92, 0x41, 0x14, 0x0a, 0x12, 0xd2, 0x01, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0xd2, 0x01, 0x08, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x73, 0x22, 0xb9, 0x06, 0x0a,
	0x07, 0x50, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x02, 0x69, 0x64, 0x12, 0x1a, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x42, 0x06, 0x92, 0x41, 0x03, 0x80, 0x01, 0x01, 0x52, 0x04,
	0x6e, 0x61, 0x6d, 0x65, 0x12, 0x21, 0x0a, 0x0c, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63,
	0x65, 0x5f, 0x69, 0x64, 0x18, 0x03, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0b, 0x77, 0x6f, 0x72, 0x6b,
	0x73, 0x70, 0x61, 0x63, 0x65, 0x49, 0x64, 0x12, 0x20, 0x0a, 0x0b, 0x64, 0x65, 0x73, 0x63, 0x72,
	0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0b, 0x64, 0x65,
	0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x12, 0x57, 0x0a, 0x1a, 0x6c, 0x61, 0x73,
	0x74, 0x5f, 0x65, 0x78, 0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x74, 0x61,
	0x72, 0x74, 0x65, 0x64, 0x5f, 0x61, 0x74, 0x18, 0x05, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e,
	0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e,
	0x54, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x52, 0x17, 0x6c, 0x61, 0x73, 0x74, 0x45,
	0x78, 0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x53, 0x74, 0x61, 0x72, 0x74, 0x65, 0x64,
	0x41, 0x74, 0x12, 0x31, 0x0a, 0x05, 0x6e, 0x6f, 0x74, 0x65, 0x73, 0x18, 0x06, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x1b, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x70,
	0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x2e, 0x76, 0x31, 0x2e, 0x4e, 0x6f, 0x74, 0x65, 0x52, 0x05,
	0x6e, 0x6f, 0x74, 0x65, 0x73, 0x12, 0x27, 0x0a, 0x0f, 0x6e, 0x75, 0x6d, 0x5f, 0x65, 0x78, 0x70,
	0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x18, 0x07, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0e,
	0x6e, 0x75, 0x6d, 0x45, 0x78, 0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x12, 0x34,
	0x0a, 0x16, 0x6e, 0x75, 0x6d, 0x5f, 0x61, 0x63, 0x74, 0x69, 0x76, 0x65, 0x5f, 0x65, 0x78, 0x70,
	0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0x18, 0x08, 0x20, 0x01, 0x28, 0x05, 0x52, 0x14,
	0x6e, 0x75, 0x6d, 0x41, 0x63, 0x74, 0x69, 0x76, 0x65, 0x45, 0x78, 0x70, 0x65, 0x72, 0x69, 0x6d,
	0x65, 0x6e, 0x74, 0x73, 0x12, 0x1a, 0x0a, 0x08, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x64,
	0x18, 0x09, 0x20, 0x01, 0x28, 0x08, 0x52, 0x08, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x64,
	0x12, 0x1a, 0x0a, 0x08, 0x75, 0x73, 0x65, 0x72, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x0a, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x08, 0x75, 0x73, 0x65, 0x72, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x1c, 0x0a, 0x09,
	0x69, 0x6d, 0x6d, 0x75, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x18, 0x0b, 0x20, 0x01, 0x28, 0x08, 0x52,
	0x09, 0x69, 0x6d, 0x6d, 0x75, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x12, 0x17, 0x0a, 0x07, 0x75, 0x73,
	0x65, 0x72, 0x5f, 0x69, 0x64, 0x18, 0x0c, 0x20, 0x01, 0x28, 0x05, 0x52, 0x06, 0x75, 0x73, 0x65,
	0x72, 0x49, 0x64, 0x12, 0x25, 0x0a, 0x0e, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63, 0x65,
	0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x0d, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0d, 0x77, 0x6f, 0x72,
	0x6b, 0x73, 0x70, 0x61, 0x63, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x3d, 0x0a, 0x05, 0x73, 0x74,
	0x61, 0x74, 0x65, 0x18, 0x0e, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x27, 0x2e, 0x64, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63, 0x65,
	0x2e, 0x76, 0x31, 0x2e, 0x57, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63, 0x65, 0x53, 0x74, 0x61,
	0x74, 0x65, 0x52, 0x05, 0x73, 0x74, 0x61, 0x74, 0x65, 0x12, 0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72,
	0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x0f, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x12, 0x10,
	0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x10, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79,
	0x12, 0x19, 0x0a, 0x08, 0x6e, 0x75, 0x6d, 0x5f, 0x72, 0x75, 0x6e, 0x73, 0x18, 0x11, 0x20, 0x01,
	0x28, 0x05, 0x52, 0x07, 0x6e, 0x75, 0x6d, 0x52, 0x75, 0x6e, 0x73, 0x3a, 0xaa, 0x01, 0x92, 0x41,
	0xa6, 0x01, 0x0a, 0xa3, 0x01, 0xd2, 0x01, 0x08, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x64,
	0xd2, 0x01, 0x0d, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65,
	0xd2, 0x01, 0x02, 0x69, 0x64, 0xd2, 0x01, 0x09, 0x69, 0x6d, 0x6d, 0x75, 0x74, 0x61, 0x62, 0x6c,
	0x65, 0xd2, 0x01, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0xd2, 0x01, 0x05, 0x6e, 0x6f, 0x74, 0x65, 0x73,
	0xd2, 0x01, 0x16, 0x6e, 0x75, 0x6d, 0x5f, 0x61, 0x63, 0x74, 0x69, 0x76, 0x65, 0x5f, 0x65, 0x78,
	0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0xd2, 0x01, 0x0f, 0x6e, 0x75, 0x6d, 0x5f,
	0x65, 0x78, 0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x73, 0xd2, 0x01, 0x08, 0x6e, 0x75,
	0x6d, 0x5f, 0x72, 0x75, 0x6e, 0x73, 0xd2, 0x01, 0x05, 0x73, 0x74, 0x61, 0x74, 0x65, 0xd2, 0x01,
	0x07, 0x75, 0x73, 0x65, 0x72, 0x5f, 0x69, 0x64, 0xd2, 0x01, 0x08, 0x75, 0x73, 0x65, 0x72, 0x6e,
	0x61, 0x6d, 0x65, 0xd2, 0x01, 0x0c, 0x77, 0x6f, 0x72, 0x6b, 0x73, 0x70, 0x61, 0x63, 0x65, 0x5f,
	0x69, 0x64, 0xd2, 0x01, 0x03, 0x6b, 0x65, 0x79, 0x22, 0xb0, 0x01, 0x0a, 0x0c, 0x50, 0x61, 0x74,
	0x63, 0x68, 0x50, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x12, 0x30, 0x0a, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1c, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72, 0x69, 0x6e, 0x67,
	0x56, 0x61, 0x6c, 0x75, 0x65, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x3e, 0x0a, 0x0b, 0x64,
	0x65, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x1c, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62,
	0x75, 0x66, 0x2e, 0x53, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x52, 0x0b,
	0x64, 0x65, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x12, 0x2e, 0x0a, 0x03, 0x6b,
	0x65, 0x79, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1c, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c,
	0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72, 0x69, 0x6e,
	0x67, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x22, 0x77, 0x0a, 0x0c, 0x4d,
	0x65, 0x74, 0x72, 0x69, 0x63, 0x73, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x12, 0x21, 0x0a, 0x0c, 0x6d,
	0x65, 0x74, 0x72, 0x69, 0x63, 0x73, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x0b, 0x6d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x73, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x10,
	0x0a, 0x03, 0x6d, 0x69, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x01, 0x52, 0x03, 0x6d, 0x69, 0x6e,
	0x12, 0x10, 0x0a, 0x03, 0x6d, 0x61, 0x78, 0x18, 0x03, 0x20, 0x01, 0x28, 0x01, 0x52, 0x03, 0x6d,
	0x61, 0x78, 0x3a, 0x20, 0x92, 0x41, 0x1d, 0x0a, 0x1b, 0xd2, 0x01, 0x0c, 0x6d, 0x65, 0x74, 0x72,
	0x69, 0x63, 0x73, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0xd2, 0x01, 0x03, 0x6d, 0x69, 0x6e, 0xd2, 0x01,
	0x03, 0x6d, 0x61, 0x78, 0x2a, 0xa8, 0x02, 0x0a, 0x0c, 0x4c, 0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x54, 0x79, 0x70, 0x65, 0x12, 0x1d, 0x0a, 0x19, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f,
	0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x55, 0x4e, 0x53, 0x50, 0x45, 0x43, 0x49, 0x46, 0x49,
	0x45, 0x44, 0x10, 0x00, 0x12, 0x1c, 0x0a, 0x18, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e,
	0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x45, 0x58, 0x50, 0x45, 0x52, 0x49, 0x4d, 0x45, 0x4e, 0x54,
	0x10, 0x01, 0x12, 0x21, 0x0a, 0x1d, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x54,
	0x59, 0x50, 0x45, 0x5f, 0x48, 0x59, 0x50, 0x45, 0x52, 0x50, 0x41, 0x52, 0x41, 0x4d, 0x45, 0x54,
	0x45, 0x52, 0x53, 0x10, 0x02, 0x12, 0x1d, 0x0a, 0x19, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f,
	0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x56, 0x41, 0x4c, 0x49, 0x44, 0x41, 0x54, 0x49, 0x4f,
	0x4e, 0x53, 0x10, 0x03, 0x12, 0x1a, 0x0a, 0x16, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e,
	0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x54, 0x52, 0x41, 0x49, 0x4e, 0x49, 0x4e, 0x47, 0x10, 0x04,
	0x12, 0x1f, 0x0a, 0x1b, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x54, 0x59, 0x50,
	0x45, 0x5f, 0x43, 0x55, 0x53, 0x54, 0x4f, 0x4d, 0x5f, 0x4d, 0x45, 0x54, 0x52, 0x49, 0x43, 0x10,
	0x05, 0x12, 0x15, 0x0a, 0x11, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x54, 0x59,
	0x50, 0x45, 0x5f, 0x52, 0x55, 0x4e, 0x10, 0x06, 0x12, 0x25, 0x0a, 0x21, 0x4c, 0x4f, 0x43, 0x41,
	0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x52, 0x55, 0x4e, 0x5f, 0x48, 0x59,
	0x50, 0x45, 0x52, 0x50, 0x41, 0x52, 0x41, 0x4d, 0x45, 0x54, 0x45, 0x52, 0x53, 0x10, 0x07, 0x12,
	0x1e, 0x0a, 0x1a, 0x4c, 0x4f, 0x43, 0x41, 0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45,
	0x5f, 0x52, 0x55, 0x4e, 0x5f, 0x4d, 0x45, 0x54, 0x41, 0x44, 0x41, 0x54, 0x41, 0x10, 0x08, 0x2a,
	0x84, 0x01, 0x0a, 0x0a, 0x43, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x54, 0x79, 0x70, 0x65, 0x12, 0x1b,
	0x0a, 0x17, 0x43, 0x4f, 0x4c, 0x55, 0x4d, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x55, 0x4e,
	0x53, 0x50, 0x45, 0x43, 0x49, 0x46, 0x49, 0x45, 0x44, 0x10, 0x00, 0x12, 0x14, 0x0a, 0x10, 0x43,
	0x4f, 0x4c, 0x55, 0x4d, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x54, 0x45, 0x58, 0x54, 0x10,
	0x01, 0x12, 0x16, 0x0a, 0x12, 0x43, 0x4f, 0x4c, 0x55, 0x4d, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45,
	0x5f, 0x4e, 0x55, 0x4d, 0x42, 0x45, 0x52, 0x10, 0x02, 0x12, 0x14, 0x0a, 0x10, 0x43, 0x4f, 0x4c,
	0x55, 0x4d, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x44, 0x41, 0x54, 0x45, 0x10, 0x03, 0x12,
	0x15, 0x0a, 0x11, 0x43, 0x4f, 0x4c, 0x55, 0x4d, 0x4e, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x41,
	0x52, 0x52, 0x41, 0x59, 0x10, 0x04, 0x42, 0x39, 0x5a, 0x37, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62,
	0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2d,
	0x61, 0x69, 0x2f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x76,
	0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_determined_project_v1_project_proto_rawDescOnce sync.Once
	file_determined_project_v1_project_proto_rawDescData = file_determined_project_v1_project_proto_rawDesc
)

func file_determined_project_v1_project_proto_rawDescGZIP() []byte {
	file_determined_project_v1_project_proto_rawDescOnce.Do(func() {
		file_determined_project_v1_project_proto_rawDescData = protoimpl.X.CompressGZIP(file_determined_project_v1_project_proto_rawDescData)
	})
	return file_determined_project_v1_project_proto_rawDescData
}

var file_determined_project_v1_project_proto_enumTypes = make([]protoimpl.EnumInfo, 2)
var file_determined_project_v1_project_proto_msgTypes = make([]protoimpl.MessageInfo, 5)
var file_determined_project_v1_project_proto_goTypes = []interface{}{
	(LocationType)(0),               // 0: determined.project.v1.LocationType
	(ColumnType)(0),                 // 1: determined.project.v1.ColumnType
	(*ProjectColumn)(nil),           // 2: determined.project.v1.ProjectColumn
	(*Note)(nil),                    // 3: determined.project.v1.Note
	(*Project)(nil),                 // 4: determined.project.v1.Project
	(*PatchProject)(nil),            // 5: determined.project.v1.PatchProject
	(*MetricsRange)(nil),            // 6: determined.project.v1.MetricsRange
	(*timestamp.Timestamp)(nil),     // 7: google.protobuf.Timestamp
	(workspacev1.WorkspaceState)(0), // 8: determined.workspace.v1.WorkspaceState
	(*wrappers.StringValue)(nil),    // 9: google.protobuf.StringValue
}
var file_determined_project_v1_project_proto_depIdxs = []int32{
	0, // 0: determined.project.v1.ProjectColumn.location:type_name -> determined.project.v1.LocationType
	1, // 1: determined.project.v1.ProjectColumn.type:type_name -> determined.project.v1.ColumnType
	7, // 2: determined.project.v1.Project.last_experiment_started_at:type_name -> google.protobuf.Timestamp
	3, // 3: determined.project.v1.Project.notes:type_name -> determined.project.v1.Note
	8, // 4: determined.project.v1.Project.state:type_name -> determined.workspace.v1.WorkspaceState
	9, // 5: determined.project.v1.PatchProject.name:type_name -> google.protobuf.StringValue
	9, // 6: determined.project.v1.PatchProject.description:type_name -> google.protobuf.StringValue
	9, // 7: determined.project.v1.PatchProject.key:type_name -> google.protobuf.StringValue
	8, // [8:8] is the sub-list for method output_type
	8, // [8:8] is the sub-list for method input_type
	8, // [8:8] is the sub-list for extension type_name
	8, // [8:8] is the sub-list for extension extendee
	0, // [0:8] is the sub-list for field type_name
}

func init() { file_determined_project_v1_project_proto_init() }
func file_determined_project_v1_project_proto_init() {
	if File_determined_project_v1_project_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_determined_project_v1_project_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProjectColumn); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_project_v1_project_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Note); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_project_v1_project_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Project); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_project_v1_project_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PatchProject); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_project_v1_project_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*MetricsRange); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_determined_project_v1_project_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   5,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_determined_project_v1_project_proto_goTypes,
		DependencyIndexes: file_determined_project_v1_project_proto_depIdxs,
		EnumInfos:         file_determined_project_v1_project_proto_enumTypes,
		MessageInfos:      file_determined_project_v1_project_proto_msgTypes,
	}.Build()
	File_determined_project_v1_project_proto = out.File
	file_determined_project_v1_project_proto_rawDesc = nil
	file_determined_project_v1_project_proto_goTypes = nil
	file_determined_project_v1_project_proto_depIdxs = nil
}
