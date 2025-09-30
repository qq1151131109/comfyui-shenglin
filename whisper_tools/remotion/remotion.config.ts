// See all configuration options: https://remotion.dev/docs/config
// Each option also is available as a CLI flag: https://remotion.dev/docs/cli

// Note: When using the Node.JS APIs, the config file doesn't apply. Instead, pass options directly to the APIs

import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("png");  // 更高质量的图像格式
Config.setOverwriteOutput(true);
Config.setConcurrency(1); // 避免并发造成的不一致
