# ComfyUI 插件：调用和执行 RunningHub 工作流

该插件用于在本地 ComfyUI 环境中便捷地调用和执行 RunningHub 上的工作流。它是对 [RunningHub API 文档](https://gold-spleen-bf1.notion.site/RunningHub-API-1432ece0cf5f8026aaa8e4b9190f6f8e) 的一个 ComfyUI 实现。在使用本插件之前建议花2分钟阅读。如果你希望扩展该插件，或在使用过程中遇到问题，请参考上述文档。
### 可以通过本插件，将RunningHub方便的接入[Photoshop](https://github.com/NimaNzrii/comfyui-photoshop)，[变现宝](https://github.com/zhulu111/ComfyUI_Bxb) 等各种插件
## 使用步骤

### 1. 安装插件
在终端中运行以下命令以克隆插件到本地：
```bash
git clone https://github.com/HM-RunningHub/ComfyUI_RH_APICall
```
### 2. 注册并获取 API Key
访问 [RunningHub 官网](https://www.runninghub.cn) 注册账户并获取你的 API Key。

### 3. 在 ComfyUI 中调用 RunningHub 上的个人工作流
完成安装和配置后，你就可以在本地的 ComfyUI 环境中调用 RunningHub 上的个人工作流了。请把示例中的相关配置参数改成你自己的。

### 4. 本地工作流与RunningHub工作流节点对应关系与配置说明。
通过NodeInfolist节点，可以修改RH工作流每个节点的值。比如常见的提示词，种子，生成批次等
![image](https://github.com/user-attachments/assets/e6d76026-13bb-4ee7-8bcf-2cbc64a046ce)

### 5. 实例(工作流json在examples目录下)
#### 文生图：
![image](https://github.com/user-attachments/assets/3b00beeb-1d0d-4fc2-b635-d31cfcf06887)
#### 图生图：
![image](https://github.com/user-attachments/assets/552bf53c-8913-474e-838a-c110e9dbc6d0)
#### 混元文生视频：
![image](https://github.com/user-attachments/assets/ed7cca06-f8cb-4eda-9dd8-c56464fd2414)
#### 连接Photoshop
![image](https://github.com/user-attachments/assets/72c7ff4a-f6ef-43d5-a95c-242fbff5aafc)
#### 对RH输出的多张图片，再做进一步灵活处理
![image](https://github.com/user-attachments/assets/d28488f4-c5e5-436d-a9fd-50f9083ef3ff)
