// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// source: determined/api/v2/api.proto

package apiv2

import (
	context "context"
	_ "github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger/options"
	_ "google.golang.org/genproto/googleapis/api/annotations"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

var File_determined_api_v2_api_proto protoreflect.FileDescriptor

var file_determined_api_v2_api_proto_rawDesc = []byte{
	0x0a, 0x1b, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x61, 0x70, 0x69,
	0x2f, 0x76, 0x32, 0x2f, 0x61, 0x70, 0x69, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x11, 0x64,
	0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32,
	0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x61, 0x6e, 0x6e,
	0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x2c,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x2d, 0x67, 0x65, 0x6e, 0x2d, 0x73, 0x77, 0x61, 0x67, 0x67,
	0x65, 0x72, 0x2f, 0x6f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74,
	0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1e, 0x64, 0x65,
	0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x32, 0x2f,
	0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x32, 0xb1, 0x09, 0x0a,
	0x0a, 0x44, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x12, 0x89, 0x01, 0x0a, 0x09,
	0x47, 0x65, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x12, 0x23, 0x2e, 0x64, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x47, 0x65,
	0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x24,
	0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e,
	0x76, 0x32, 0x2e, 0x47, 0x65, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x31, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61, 0x72, 0x63,
	0x68, 0x65, 0x73, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x1e, 0x12, 0x1c, 0x2f, 0x61, 0x70, 0x69, 0x2f,
	0x76, 0x32, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x2f, 0x7b, 0x73, 0x65, 0x61,
	0x72, 0x63, 0x68, 0x5f, 0x69, 0x64, 0x7d, 0x12, 0x8c, 0x01, 0x0a, 0x0d, 0x47, 0x65, 0x74, 0x53,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x73, 0x12, 0x27, 0x2e, 0x64, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x47, 0x65,
	0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x1a, 0x28, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e,
	0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x47, 0x65, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68,
	0x54, 0x61, 0x67, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x28, 0x92, 0x41,
	0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x82, 0xd3, 0xe4, 0x93, 0x02,
	0x15, 0x12, 0x13, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x32, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63,
	0x68, 0x2f, 0x74, 0x61, 0x67, 0x73, 0x12, 0x9d, 0x01, 0x0a, 0x0c, 0x50, 0x75, 0x74, 0x53, 0x65,
	0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x12, 0x26, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d,
	0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x75, 0x74, 0x53,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a,
	0x27, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69,
	0x2e, 0x76, 0x32, 0x2e, 0x50, 0x75, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x3c, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x29, 0x1a, 0x27, 0x2f,
	0x61, 0x70, 0x69, 0x2f, 0x76, 0x32, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x2f,
	0x7b, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x74, 0x61, 0x67, 0x73,
	0x2f, 0x7b, 0x74, 0x61, 0x67, 0x7d, 0x12, 0xa6, 0x01, 0x0a, 0x0f, 0x44, 0x65, 0x6c, 0x65, 0x74,
	0x65, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x12, 0x29, 0x2e, 0x64, 0x65, 0x74,
	0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x44,
	0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x2a, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e,
	0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65,
	0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x54, 0x61, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x22, 0x3c, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73,
	0x82, 0xd3, 0xe4, 0x93, 0x02, 0x29, 0x2a, 0x27, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x31, 0x2f,
	0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x2f, 0x7b, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68,
	0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x74, 0x61, 0x67, 0x73, 0x2f, 0x7b, 0x74, 0x61, 0x67, 0x7d, 0x12,
	0xb7, 0x01, 0x0a, 0x13, 0x47, 0x65, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x45,
	0x76, 0x65, 0x6e, 0x74, 0x73, 0x56, 0x32, 0x12, 0x2d, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d,
	0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x47, 0x65, 0x74, 0x53,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x73, 0x56, 0x32, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x2e, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69,
	0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x47, 0x65, 0x74, 0x53, 0x65,
	0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x73, 0x56, 0x32, 0x52, 0x65,
	0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x41, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61,
	0x72, 0x63, 0x68, 0x65, 0x73, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x2e, 0x12, 0x2c, 0x2f, 0x61, 0x70,
	0x69, 0x2f, 0x76, 0x32, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x2f, 0x7b, 0x73,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68,
	0x65, 0x72, 0x5f, 0x65, 0x76, 0x65, 0x6e, 0x74, 0x73, 0x12, 0xcd, 0x01, 0x0a, 0x18, 0x50, 0x6f,
	0x73, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x4f, 0x70, 0x65, 0x72, 0x61, 0x74,
	0x69, 0x6f, 0x6e, 0x73, 0x56, 0x32, 0x12, 0x32, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69,
	0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x6f, 0x73, 0x74, 0x53,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x4f, 0x70, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e,
	0x73, 0x56, 0x32, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x33, 0x2e, 0x64, 0x65, 0x74,
	0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x50,
	0x6f, 0x73, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x4f, 0x70, 0x65, 0x72, 0x61,
	0x74, 0x69, 0x6f, 0x6e, 0x73, 0x56, 0x32, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x48, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x82, 0xd3,
	0xe4, 0x93, 0x02, 0x35, 0x22, 0x30, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x32, 0x2f, 0x73, 0x65,
	0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x2f, 0x7b, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x5f, 0x69,
	0x64, 0x7d, 0x2f, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72, 0x5f, 0x6f, 0x70, 0x65, 0x72,
	0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x3a, 0x01, 0x2a, 0x12, 0xb4, 0x01, 0x0a, 0x13, 0x50, 0x75,
	0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x52, 0x65, 0x74, 0x61, 0x69, 0x6e, 0x4c, 0x6f, 0x67,
	0x73, 0x12, 0x2d, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61,
	0x70, 0x69, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x75, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x52,
	0x65, 0x74, 0x61, 0x69, 0x6e, 0x4c, 0x6f, 0x67, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74,
	0x1a, 0x2e, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70,
	0x69, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x75, 0x74, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x52, 0x65,
	0x74, 0x61, 0x69, 0x6e, 0x4c, 0x6f, 0x67, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x22, 0x3e, 0x92, 0x41, 0x0a, 0x0a, 0x08, 0x53, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x73, 0x82,
	0xd3, 0xe4, 0x93, 0x02, 0x2b, 0x1a, 0x26, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x76, 0x31, 0x2f, 0x73,
	0x65, 0x61, 0x72, 0x63, 0x68, 0x2f, 0x7b, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x5f, 0x69, 0x64,
	0x7d, 0x2f, 0x72, 0x65, 0x74, 0x61, 0x69, 0x6e, 0x5f, 0x6c, 0x6f, 0x67, 0x73, 0x3a, 0x01, 0x2a,
	0x42, 0xda, 0x07, 0x5a, 0x33, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f,
	0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2d, 0x61, 0x69, 0x2f, 0x64, 0x65,
	0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x70,
	0x6b, 0x67, 0x2f, 0x61, 0x70, 0x69, 0x76, 0x32, 0x92, 0x41, 0xa1, 0x07, 0x12, 0x95, 0x06, 0x0a,
	0x15, 0x44, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x20, 0x41, 0x50, 0x49, 0x20,
	0x28, 0x42, 0x65, 0x74, 0x61, 0x29, 0x12, 0xf5, 0x04, 0x44, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69,
	0x6e, 0x65, 0x64, 0x20, 0x68, 0x65, 0x6c, 0x70, 0x73, 0x20, 0x64, 0x65, 0x65, 0x70, 0x20, 0x6c,
	0x65, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x20, 0x74, 0x65, 0x61, 0x6d, 0x73, 0x20, 0x74, 0x72,
	0x61, 0x69, 0x6e, 0x20, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x20, 0x6d, 0x6f, 0x72, 0x65, 0x20,
	0x71, 0x75, 0x69, 0x63, 0x6b, 0x6c, 0x79, 0x2c, 0x20, 0x65, 0x61, 0x73, 0x69, 0x6c, 0x79, 0x20,
	0x73, 0x68, 0x61, 0x72, 0x65, 0x20, 0x47, 0x50, 0x55, 0x20, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72,
	0x63, 0x65, 0x73, 0x2c, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x65, 0x66, 0x66, 0x65, 0x63, 0x74, 0x69,
	0x76, 0x65, 0x6c, 0x79, 0x20, 0x63, 0x6f, 0x6c, 0x6c, 0x61, 0x62, 0x6f, 0x72, 0x61, 0x74, 0x65,
	0x2e, 0x20, 0x44, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x20, 0x61, 0x6c, 0x6c,
	0x6f, 0x77, 0x73, 0x20, 0x64, 0x65, 0x65, 0x70, 0x20, 0x6c, 0x65, 0x61, 0x72, 0x6e, 0x69, 0x6e,
	0x67, 0x20, 0x65, 0x6e, 0x67, 0x69, 0x6e, 0x65, 0x65, 0x72, 0x73, 0x20, 0x74, 0x6f, 0x20, 0x66,
	0x6f, 0x63, 0x75, 0x73, 0x20, 0x6f, 0x6e, 0x20, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x69, 0x6e, 0x67,
	0x20, 0x61, 0x6e, 0x64, 0x20, 0x74, 0x72, 0x61, 0x69, 0x6e, 0x69, 0x6e, 0x67, 0x20, 0x6d, 0x6f,
	0x64, 0x65, 0x6c, 0x73, 0x20, 0x61, 0x74, 0x20, 0x73, 0x63, 0x61, 0x6c, 0x65, 0x2c, 0x20, 0x77,
	0x69, 0x74, 0x68, 0x6f, 0x75, 0x74, 0x20, 0x6e, 0x65, 0x65, 0x64, 0x69, 0x6e, 0x67, 0x20, 0x74,
	0x6f, 0x20, 0x77, 0x6f, 0x72, 0x72, 0x79, 0x20, 0x61, 0x62, 0x6f, 0x75, 0x74, 0x20, 0x44, 0x65,
	0x76, 0x4f, 0x70, 0x73, 0x20, 0x6f, 0x72, 0x20, 0x77, 0x72, 0x69, 0x74, 0x69, 0x6e, 0x67, 0x20,
	0x63, 0x75, 0x73, 0x74, 0x6f, 0x6d, 0x20, 0x63, 0x6f, 0x64, 0x65, 0x20, 0x66, 0x6f, 0x72, 0x20,
	0x63, 0x6f, 0x6d, 0x6d, 0x6f, 0x6e, 0x20, 0x74, 0x61, 0x73, 0x6b, 0x73, 0x20, 0x6c, 0x69, 0x6b,
	0x65, 0x20, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x20, 0x74, 0x6f, 0x6c, 0x65, 0x72, 0x61, 0x6e, 0x63,
	0x65, 0x20, 0x6f, 0x72, 0x20, 0x65, 0x78, 0x70, 0x65, 0x72, 0x69, 0x6d, 0x65, 0x6e, 0x74, 0x20,
	0x74, 0x72, 0x61, 0x63, 0x6b, 0x69, 0x6e, 0x67, 0x2e, 0x0a, 0x0a, 0x59, 0x6f, 0x75, 0x20, 0x63,
	0x61, 0x6e, 0x20, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x20, 0x6f, 0x66, 0x20, 0x44, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x20, 0x61, 0x73, 0x20, 0x61, 0x20, 0x70, 0x6c, 0x61, 0x74,
	0x66, 0x6f, 0x72, 0x6d, 0x20, 0x74, 0x68, 0x61, 0x74, 0x20, 0x62, 0x72, 0x69, 0x64, 0x67, 0x65,
	0x73, 0x20, 0x74, 0x68, 0x65, 0x20, 0x67, 0x61, 0x70, 0x20, 0x62, 0x65, 0x74, 0x77, 0x65, 0x65,
	0x6e, 0x20, 0x74, 0x6f, 0x6f, 0x6c, 0x73, 0x20, 0x6c, 0x69, 0x6b, 0x65, 0x20, 0x54, 0x65, 0x6e,
	0x73, 0x6f, 0x72, 0x46, 0x6c, 0x6f, 0x77, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x50, 0x79, 0x54, 0x6f,
	0x72, 0x63, 0x68, 0x20, 0x2d, 0x2d, 0x2d, 0x20, 0x77, 0x68, 0x69, 0x63, 0x68, 0x20, 0x77, 0x6f,
	0x72, 0x6b, 0x20, 0x67, 0x72, 0x65, 0x61, 0x74, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x61, 0x20, 0x73,
	0x69, 0x6e, 0x67, 0x6c, 0x65, 0x20, 0x72, 0x65, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x65, 0x72,
	0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x61, 0x20, 0x73, 0x69, 0x6e, 0x67, 0x6c, 0x65, 0x20, 0x47,
	0x50, 0x55, 0x20, 0x2d, 0x2d, 0x2d, 0x20, 0x74, 0x6f, 0x20, 0x74, 0x68, 0x65, 0x20, 0x63, 0x68,
	0x61, 0x6c, 0x6c, 0x65, 0x6e, 0x67, 0x65, 0x73, 0x20, 0x74, 0x68, 0x61, 0x74, 0x20, 0x61, 0x72,
	0x69, 0x73, 0x65, 0x20, 0x77, 0x68, 0x65, 0x6e, 0x20, 0x64, 0x6f, 0x69, 0x6e, 0x67, 0x20, 0x64,
	0x65, 0x65, 0x70, 0x20, 0x6c, 0x65, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x20, 0x61, 0x74, 0x20,
	0x73, 0x63, 0x61, 0x6c, 0x65, 0x2c, 0x20, 0x61, 0x73, 0x20, 0x74, 0x65, 0x61, 0x6d, 0x73, 0x2c,
	0x20, 0x63, 0x6c, 0x75, 0x73, 0x74, 0x65, 0x72, 0x73, 0x2c, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x64,
	0x61, 0x74, 0x61, 0x20, 0x73, 0x65, 0x74, 0x73, 0x20, 0x61, 0x6c, 0x6c, 0x20, 0x69, 0x6e, 0x63,
	0x72, 0x65, 0x61, 0x73, 0x65, 0x20, 0x69, 0x6e, 0x20, 0x73, 0x69, 0x7a, 0x65, 0x2e, 0x22, 0x40,
	0x0a, 0x0d, 0x44, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x20, 0x41, 0x49, 0x12,
	0x16, 0x68, 0x74, 0x74, 0x70, 0x73, 0x3a, 0x2f, 0x2f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69,
	0x6e, 0x65, 0x64, 0x2e, 0x61, 0x69, 0x2f, 0x1a, 0x17, 0x63, 0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69,
	0x74, 0x79, 0x40, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x69,
	0x2a, 0x3d, 0x0a, 0x0a, 0x41, 0x70, 0x61, 0x63, 0x68, 0x65, 0x20, 0x32, 0x2e, 0x30, 0x12, 0x2f,
	0x68, 0x74, 0x74, 0x70, 0x3a, 0x2f, 0x2f, 0x77, 0x77, 0x77, 0x2e, 0x61, 0x70, 0x61, 0x63, 0x68,
	0x65, 0x2e, 0x6f, 0x72, 0x67, 0x2f, 0x6c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x73, 0x2f, 0x4c,
	0x49, 0x43, 0x45, 0x4e, 0x53, 0x45, 0x2d, 0x32, 0x2e, 0x30, 0x2e, 0x68, 0x74, 0x6d, 0x6c, 0x32,
	0x03, 0x30, 0x2e, 0x32, 0x2a, 0x02, 0x01, 0x02, 0x5a, 0x4a, 0x0a, 0x48, 0x0a, 0x0b, 0x42, 0x65,
	0x61, 0x72, 0x65, 0x72, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12, 0x39, 0x08, 0x02, 0x12, 0x24, 0x42,
	0x65, 0x61, 0x72, 0x65, 0x72, 0x20, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x20, 0x61, 0x75, 0x74, 0x68,
	0x65, 0x6e, 0x74, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x20, 0x73, 0x74, 0x72, 0x61, 0x74,
	0x65, 0x67, 0x79, 0x1a, 0x0d, 0x41, 0x75, 0x74, 0x68, 0x6f, 0x72, 0x69, 0x7a, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x20, 0x02, 0x62, 0x11, 0x0a, 0x0f, 0x0a, 0x0b, 0x42, 0x65, 0x61, 0x72, 0x65, 0x72,
	0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12, 0x00, 0x72, 0x24, 0x0a, 0x1b, 0x44, 0x65, 0x74, 0x65, 0x72,
	0x6d, 0x69, 0x6e, 0x65, 0x64, 0x20, 0x41, 0x49, 0x20, 0x44, 0x6f, 0x63, 0x75, 0x6d, 0x65, 0x6e,
	0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x12, 0x05, 0x2f, 0x64, 0x6f, 0x63, 0x73, 0x62, 0x06, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var file_determined_api_v2_api_proto_goTypes = []interface{}{
	(*GetSearchRequest)(nil),                 // 0: determined.api.v2.GetSearchRequest
	(*GetSearchTagsRequest)(nil),             // 1: determined.api.v2.GetSearchTagsRequest
	(*PutSearchTagRequest)(nil),              // 2: determined.api.v2.PutSearchTagRequest
	(*DeleteSearchTagRequest)(nil),           // 3: determined.api.v2.DeleteSearchTagRequest
	(*GetSearcherEventsV2Request)(nil),       // 4: determined.api.v2.GetSearcherEventsV2Request
	(*PostSearcherOperationsV2Request)(nil),  // 5: determined.api.v2.PostSearcherOperationsV2Request
	(*PutSearchRetainLogsRequest)(nil),       // 6: determined.api.v2.PutSearchRetainLogsRequest
	(*GetSearchResponse)(nil),                // 7: determined.api.v2.GetSearchResponse
	(*GetSearchTagsResponse)(nil),            // 8: determined.api.v2.GetSearchTagsResponse
	(*PutSearchTagResponse)(nil),             // 9: determined.api.v2.PutSearchTagResponse
	(*DeleteSearchTagResponse)(nil),          // 10: determined.api.v2.DeleteSearchTagResponse
	(*GetSearcherEventsV2Response)(nil),      // 11: determined.api.v2.GetSearcherEventsV2Response
	(*PostSearcherOperationsV2Response)(nil), // 12: determined.api.v2.PostSearcherOperationsV2Response
	(*PutSearchRetainLogsResponse)(nil),      // 13: determined.api.v2.PutSearchRetainLogsResponse
}
var file_determined_api_v2_api_proto_depIdxs = []int32{
	0,  // 0: determined.api.v2.Determined.GetSearch:input_type -> determined.api.v2.GetSearchRequest
	1,  // 1: determined.api.v2.Determined.GetSearchTags:input_type -> determined.api.v2.GetSearchTagsRequest
	2,  // 2: determined.api.v2.Determined.PutSearchTag:input_type -> determined.api.v2.PutSearchTagRequest
	3,  // 3: determined.api.v2.Determined.DeleteSearchTag:input_type -> determined.api.v2.DeleteSearchTagRequest
	4,  // 4: determined.api.v2.Determined.GetSearcherEventsV2:input_type -> determined.api.v2.GetSearcherEventsV2Request
	5,  // 5: determined.api.v2.Determined.PostSearcherOperationsV2:input_type -> determined.api.v2.PostSearcherOperationsV2Request
	6,  // 6: determined.api.v2.Determined.PutSearchRetainLogs:input_type -> determined.api.v2.PutSearchRetainLogsRequest
	7,  // 7: determined.api.v2.Determined.GetSearch:output_type -> determined.api.v2.GetSearchResponse
	8,  // 8: determined.api.v2.Determined.GetSearchTags:output_type -> determined.api.v2.GetSearchTagsResponse
	9,  // 9: determined.api.v2.Determined.PutSearchTag:output_type -> determined.api.v2.PutSearchTagResponse
	10, // 10: determined.api.v2.Determined.DeleteSearchTag:output_type -> determined.api.v2.DeleteSearchTagResponse
	11, // 11: determined.api.v2.Determined.GetSearcherEventsV2:output_type -> determined.api.v2.GetSearcherEventsV2Response
	12, // 12: determined.api.v2.Determined.PostSearcherOperationsV2:output_type -> determined.api.v2.PostSearcherOperationsV2Response
	13, // 13: determined.api.v2.Determined.PutSearchRetainLogs:output_type -> determined.api.v2.PutSearchRetainLogsResponse
	7,  // [7:14] is the sub-list for method output_type
	0,  // [0:7] is the sub-list for method input_type
	0,  // [0:0] is the sub-list for extension type_name
	0,  // [0:0] is the sub-list for extension extendee
	0,  // [0:0] is the sub-list for field type_name
}

func init() { file_determined_api_v2_api_proto_init() }
func file_determined_api_v2_api_proto_init() {
	if File_determined_api_v2_api_proto != nil {
		return
	}
	file_determined_api_v2_search_proto_init()
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_determined_api_v2_api_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   0,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_determined_api_v2_api_proto_goTypes,
		DependencyIndexes: file_determined_api_v2_api_proto_depIdxs,
	}.Build()
	File_determined_api_v2_api_proto = out.File
	file_determined_api_v2_api_proto_rawDesc = nil
	file_determined_api_v2_api_proto_goTypes = nil
	file_determined_api_v2_api_proto_depIdxs = nil
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConnInterface

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion6

// DeterminedClient is the client API for Determined service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type DeterminedClient interface {
	// Get the requested search.
	GetSearch(ctx context.Context, in *GetSearchRequest, opts ...grpc.CallOption) (*GetSearchResponse, error)
	// Get a list of unique search tags.
	GetSearchTags(ctx context.Context, in *GetSearchTagsRequest, opts ...grpc.CallOption) (*GetSearchTagsResponse, error)
	// Put a new tag on the search.
	PutSearchTag(ctx context.Context, in *PutSearchTagRequest, opts ...grpc.CallOption) (*PutSearchTagResponse, error)
	// Delete a tag from the search.
	DeleteSearchTag(ctx context.Context, in *DeleteSearchTagRequest, opts ...grpc.CallOption) (*DeleteSearchTagResponse, error)
	// Get the list of searcher events.
	GetSearcherEventsV2(ctx context.Context, in *GetSearcherEventsV2Request, opts ...grpc.CallOption) (*GetSearcherEventsV2Response, error)
	// Submit operations to a custom searcher.
	PostSearcherOperationsV2(ctx context.Context, in *PostSearcherOperationsV2Request, opts ...grpc.CallOption) (*PostSearcherOperationsV2Response, error)
	// Retain logs for a search.
	PutSearchRetainLogs(ctx context.Context, in *PutSearchRetainLogsRequest, opts ...grpc.CallOption) (*PutSearchRetainLogsResponse, error)
}

type determinedClient struct {
	cc grpc.ClientConnInterface
}

func NewDeterminedClient(cc grpc.ClientConnInterface) DeterminedClient {
	return &determinedClient{cc}
}

func (c *determinedClient) GetSearch(ctx context.Context, in *GetSearchRequest, opts ...grpc.CallOption) (*GetSearchResponse, error) {
	out := new(GetSearchResponse)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/GetSearch", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) GetSearchTags(ctx context.Context, in *GetSearchTagsRequest, opts ...grpc.CallOption) (*GetSearchTagsResponse, error) {
	out := new(GetSearchTagsResponse)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/GetSearchTags", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) PutSearchTag(ctx context.Context, in *PutSearchTagRequest, opts ...grpc.CallOption) (*PutSearchTagResponse, error) {
	out := new(PutSearchTagResponse)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/PutSearchTag", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) DeleteSearchTag(ctx context.Context, in *DeleteSearchTagRequest, opts ...grpc.CallOption) (*DeleteSearchTagResponse, error) {
	out := new(DeleteSearchTagResponse)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/DeleteSearchTag", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) GetSearcherEventsV2(ctx context.Context, in *GetSearcherEventsV2Request, opts ...grpc.CallOption) (*GetSearcherEventsV2Response, error) {
	out := new(GetSearcherEventsV2Response)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/GetSearcherEventsV2", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) PostSearcherOperationsV2(ctx context.Context, in *PostSearcherOperationsV2Request, opts ...grpc.CallOption) (*PostSearcherOperationsV2Response, error) {
	out := new(PostSearcherOperationsV2Response)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/PostSearcherOperationsV2", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *determinedClient) PutSearchRetainLogs(ctx context.Context, in *PutSearchRetainLogsRequest, opts ...grpc.CallOption) (*PutSearchRetainLogsResponse, error) {
	out := new(PutSearchRetainLogsResponse)
	err := c.cc.Invoke(ctx, "/determined.api.v2.Determined/PutSearchRetainLogs", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DeterminedServer is the server API for Determined service.
type DeterminedServer interface {
	// Get the requested search.
	GetSearch(context.Context, *GetSearchRequest) (*GetSearchResponse, error)
	// Get a list of unique search tags.
	GetSearchTags(context.Context, *GetSearchTagsRequest) (*GetSearchTagsResponse, error)
	// Put a new tag on the search.
	PutSearchTag(context.Context, *PutSearchTagRequest) (*PutSearchTagResponse, error)
	// Delete a tag from the search.
	DeleteSearchTag(context.Context, *DeleteSearchTagRequest) (*DeleteSearchTagResponse, error)
	// Get the list of searcher events.
	GetSearcherEventsV2(context.Context, *GetSearcherEventsV2Request) (*GetSearcherEventsV2Response, error)
	// Submit operations to a custom searcher.
	PostSearcherOperationsV2(context.Context, *PostSearcherOperationsV2Request) (*PostSearcherOperationsV2Response, error)
	// Retain logs for a search.
	PutSearchRetainLogs(context.Context, *PutSearchRetainLogsRequest) (*PutSearchRetainLogsResponse, error)
}

// UnimplementedDeterminedServer can be embedded to have forward compatible implementations.
type UnimplementedDeterminedServer struct {
}

func (*UnimplementedDeterminedServer) GetSearch(context.Context, *GetSearchRequest) (*GetSearchResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetSearch not implemented")
}
func (*UnimplementedDeterminedServer) GetSearchTags(context.Context, *GetSearchTagsRequest) (*GetSearchTagsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetSearchTags not implemented")
}
func (*UnimplementedDeterminedServer) PutSearchTag(context.Context, *PutSearchTagRequest) (*PutSearchTagResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method PutSearchTag not implemented")
}
func (*UnimplementedDeterminedServer) DeleteSearchTag(context.Context, *DeleteSearchTagRequest) (*DeleteSearchTagResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method DeleteSearchTag not implemented")
}
func (*UnimplementedDeterminedServer) GetSearcherEventsV2(context.Context, *GetSearcherEventsV2Request) (*GetSearcherEventsV2Response, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetSearcherEventsV2 not implemented")
}
func (*UnimplementedDeterminedServer) PostSearcherOperationsV2(context.Context, *PostSearcherOperationsV2Request) (*PostSearcherOperationsV2Response, error) {
	return nil, status.Errorf(codes.Unimplemented, "method PostSearcherOperationsV2 not implemented")
}
func (*UnimplementedDeterminedServer) PutSearchRetainLogs(context.Context, *PutSearchRetainLogsRequest) (*PutSearchRetainLogsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method PutSearchRetainLogs not implemented")
}

func RegisterDeterminedServer(s *grpc.Server, srv DeterminedServer) {
	s.RegisterService(&_Determined_serviceDesc, srv)
}

func _Determined_GetSearch_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetSearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).GetSearch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/GetSearch",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).GetSearch(ctx, req.(*GetSearchRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_GetSearchTags_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetSearchTagsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).GetSearchTags(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/GetSearchTags",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).GetSearchTags(ctx, req.(*GetSearchTagsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_PutSearchTag_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PutSearchTagRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).PutSearchTag(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/PutSearchTag",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).PutSearchTag(ctx, req.(*PutSearchTagRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_DeleteSearchTag_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DeleteSearchTagRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).DeleteSearchTag(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/DeleteSearchTag",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).DeleteSearchTag(ctx, req.(*DeleteSearchTagRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_GetSearcherEventsV2_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetSearcherEventsV2Request)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).GetSearcherEventsV2(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/GetSearcherEventsV2",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).GetSearcherEventsV2(ctx, req.(*GetSearcherEventsV2Request))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_PostSearcherOperationsV2_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PostSearcherOperationsV2Request)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).PostSearcherOperationsV2(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/PostSearcherOperationsV2",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).PostSearcherOperationsV2(ctx, req.(*PostSearcherOperationsV2Request))
	}
	return interceptor(ctx, in, info, handler)
}

func _Determined_PutSearchRetainLogs_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PutSearchRetainLogsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DeterminedServer).PutSearchRetainLogs(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/determined.api.v2.Determined/PutSearchRetainLogs",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DeterminedServer).PutSearchRetainLogs(ctx, req.(*PutSearchRetainLogsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _Determined_serviceDesc = grpc.ServiceDesc{
	ServiceName: "determined.api.v2.Determined",
	HandlerType: (*DeterminedServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetSearch",
			Handler:    _Determined_GetSearch_Handler,
		},
		{
			MethodName: "GetSearchTags",
			Handler:    _Determined_GetSearchTags_Handler,
		},
		{
			MethodName: "PutSearchTag",
			Handler:    _Determined_PutSearchTag_Handler,
		},
		{
			MethodName: "DeleteSearchTag",
			Handler:    _Determined_DeleteSearchTag_Handler,
		},
		{
			MethodName: "GetSearcherEventsV2",
			Handler:    _Determined_GetSearcherEventsV2_Handler,
		},
		{
			MethodName: "PostSearcherOperationsV2",
			Handler:    _Determined_PostSearcherOperationsV2_Handler,
		},
		{
			MethodName: "PutSearchRetainLogs",
			Handler:    _Determined_PutSearchRetainLogs_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "determined/api/v2/api.proto",
}
