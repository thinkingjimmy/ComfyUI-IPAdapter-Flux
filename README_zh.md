# ComfyUI-IPAdapter-Flux

[Read this in English](./README.md)

<div align="center">
<img src=logo.jpeg width="25%"/>
</div>
<p align="center">
 👋 加入我们的 <a href="https://discord.gg/5TuxSjJya6" target="_blank">Discord</a> 
</p>
<p align="center">
 📍 前往<a href="https://www.shakker.ai/shakker-generator"> shakker-generator </a> 和 <a href="https://www.shakker.ai/online-comfyui"> Online ComfyUI</a> 体验在线的FLUX.1-dev-IP-Adapter。
</p>

## 重要提示

请升级到 ComfyUI commit `2fd9c13` 之后的版本。  

## 项目更新

- **update**: ```2024/12/10```: 支持多个ipadapter组合，感谢<a href="https://github.com/Slickytail">Slickytail。
- **update**: ```2024/11/25```: 适配comfyui最新版本，在comfyui commit `2fd9c13` 上测试，可以正常加卸载权重。
- 🌱 **Source**: ```2024/11/22```: 我们开源了FLUX.1-dev-IP-Adapter，这是基于FLUX.1 dev的IPAdapter模型，您可以访问 [ipadapter 权重](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) 。

## 快速开始

### 安装

1. 前往 `ComfyUI/custom_nodes`  
2. 克隆此仓库，路径应为 `ComfyUI/custom_nodes/comfyui-ipadapter-flux/*`，其中 `*` 表示仓库中的所有文件。  
3. 进入 `ComfyUI/custom_nodes/comfyui-ipadapter-flux/` 并运行 `pip install -r requirements.txt`。  
4. 下载 [ipadapter 权重](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) 到 `ComfyUI/models/ipadapter-flux`。  
5. 安装完成后运行 ComfyUI！  

### 运行工作流

[参考工作流](./workflows/ipadapter_example.json)

<div align="center">
<img src=./workflows/ipadapter_example.png width="100%"/>
</div>

[调节开始和结束步数](./workflows/ipadapter_example_start_end_percent.json)

<div align="center">
<img src=./workflows/ipadapter_example_start_end_percent.png width="100%"/>
</div>

[使用多张参考图](./workflows/multi-ipadapter_example.json)

<div align="center">
<img src=./workflows/multi-ipadapter_example.png width="100%"/>
</div>

### 在线体验

您可以使用[shakker-generator](https://www.shakker.ai/shakker-generator)和[Online ComfyUI](https://www.shakker.ai/online-comfyui)体验在线的FLUX.1-dev-IP-Adapter

## 模型协议

本仓库代码使用 [Apache 2.0 协议](./LICENSE) 发布。

FLUX.1-dev-IP-Adapter 模型
根据 [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
许可证发布。