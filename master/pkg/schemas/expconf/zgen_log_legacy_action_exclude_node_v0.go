// Code generated by gen.py. DO NOT EDIT.

package expconf

import (
	"github.com/santhosh-tekuri/jsonschema/v2"

	"github.com/determined-ai/determined/master/pkg/schemas"
)

func (l LogLegacyActionExcludeNodeV0) ParsedSchema() interface{} {
	return schemas.ParsedLogLegacyActionExcludeNodeV0()
}

func (l LogLegacyActionExcludeNodeV0) SanityValidator() *jsonschema.Schema {
	return schemas.GetSanityValidator("http://determined.ai/schemas/expconf/v0/log-legacy-action-exclude-node.json")
}

func (l LogLegacyActionExcludeNodeV0) CompletenessValidator() *jsonschema.Schema {
	return schemas.GetCompletenessValidator("http://determined.ai/schemas/expconf/v0/log-legacy-action-exclude-node.json")
}
