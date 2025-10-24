# LM Studio Troubleshooting Guide

## Common Issues

### Model Name Error
- **Problem**: "Model not found" error
- **Solution**: Use model name without prefix (e.g., `"gpt-oss-20b"` not `"openai/gpt-oss-20b"`)
- **Available Models**: Check error message for list of available models

### Connection Issues
- **LM Studio URL**: http://192.168.2.2:1234/v1/chat/completions
- **Requirements**: 
  - LM Studio must be running
  - Model must be loaded
  - Server must be accessible on network

### Testing Script
Located at: Week02/test_lm_studio.py
- Tests connection and validates response structure
- Shows full JSON response for debugging