/**
 * 视频裁切界面 - 简化版本
 */

import { app } from "../../scripts/app.js";

console.log("🎬 视频裁切扩展开始加载...");

// MD5哈希计算函数，使用Web Crypto API（如果支持的话）或fallback算法
async function calculateMD5Hash(str) {
    try {
        // 尝试使用Web Crypto API的SHA-256（更兼容）
        const encoder = new TextEncoder();
        const data = encoder.encode(str);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        return hashHex.substring(0, 8); // 只取前8位
    } catch (error) {
        // Fallback: 使用简单但稳定的哈希算法
        console.log('⚠️ Web Crypto API不可用，使用fallback哈希算法');
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        const hex = Math.abs(hash).toString(16);
        return hex.padStart(8, '0').substring(0, 8);
    }
}

// 宽高比预设
const RATIOS = {
    "16:9": [16, 9],
    "9:16": [9, 16],
    "1:1": [1, 1],
    "4:3": [4, 3]
};

// 创建比例选择器
function createRatioSelector(node) {
    return {
        name: "ratio_selector",
        type: "ratio_buttons",
        value: "",

        draw: function(ctx, node, widgetWidth, y, widgetHeight) {
            const margin = 10;
            const buttonHeight = 30;
            const ratioKeys = Object.keys(RATIOS);
            const buttonWidth = (widgetWidth - margin * 2 - (ratioKeys.length - 1) * 5) / ratioKeys.length;

            ctx.font = "12px Arial";
            ctx.textAlign = "center";

            ratioKeys.forEach((ratio, i) => {
                const x = margin + i * (buttonWidth + 5);

                // 获取当前选择的比例
                const currentRatio = this.getParam(node, "aspect_ratio") || "";
                const isActive = currentRatio === ratio;

                // 绘制按钮
                ctx.fillStyle = isActive ? "#4CAF50" : "#666";
                ctx.fillRect(x, y, buttonWidth, buttonHeight);

                ctx.strokeStyle = "#999";
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, buttonWidth, buttonHeight);

                // 绘制文本
                ctx.fillStyle = "#ffffff";
                ctx.fillText(ratio, x + buttonWidth/2, y + buttonHeight/2 + 4);
            });

            return buttonHeight + 15;
        },

        mouse: function(event, pos, node) {
            if (event.type === "pointerdown") {
                const margin = 10;
                const ratioKeys = Object.keys(RATIOS);
                const buttonWidth = (node.size[0] - margin * 2 - (ratioKeys.length - 1) * 5) / ratioKeys.length;

                for (let i = 0; i < ratioKeys.length; i++) {
                    const x = margin + i * (buttonWidth + 5);
                    if (pos[0] >= x && pos[0] <= x + buttonWidth) {
                        this.applyRatio(ratioKeys[i], node);
                        return true;
                    }
                }
            }
            return false;
        },

        getParam: function(node, name) {
            const widget = node.widgets.find(w => w.name === name);
            return widget ? widget.value : null;
        },

        setParam: function(node, name, value) {
            const widget = node.widgets.find(w => w.name === name);
            if (widget) {
                widget.value = value;
                console.log(`设置参数 ${name} = ${value}`);
            }
        },

        applyRatio: function(ratio, node) {
            console.log(`应用比例: ${ratio}`);

            const [w, h] = RATIOS[ratio];
            const targetAspect = w / h;

            const videoWidth = node.videoWidth || 1920;
            const videoHeight = node.videoHeight || 1080;

            let cropWidth, cropHeight;

            if (videoWidth / videoHeight > targetAspect) {
                cropHeight = videoHeight;
                cropWidth = Math.round(cropHeight * targetAspect);
            } else {
                cropWidth = videoWidth;
                cropHeight = Math.round(cropWidth / targetAspect);
            }

            const cropX = Math.round((videoWidth - cropWidth) / 2);
            const cropY = Math.round((videoHeight - cropHeight) / 2);

            // 更新参数
            this.setParam(node, "pos_x", cropX);
            this.setParam(node, "pos_y", cropY);
            this.setParam(node, "crop_width", cropWidth);
            this.setParam(node, "crop_height", cropHeight);
            this.setParam(node, "aspect_ratio", ratio);
        },

        computeSize: function(width) {
            return [width, 45];
        }
    };
}

// 创建预览界面
function createPreview(node) {
    return {
        name: "crop_preview",
        type: "crop_preview",
        value: "",
        options: { serialize: false },
        serialize: false,

        // 计算智能居中模式的裁切坐标
        calculateCropCoordinates: function(node, aspectRatio, offsetX = 0, offsetY = 0) {
            const videoWidth = node.videoWidth || 1920;
            const videoHeight = node.videoHeight || 1080;

            const RATIOS = {
                "16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1], "4:3": [4, 3],
                "3:4": [3, 4], "21:9": [21, 9], "2:1": [2, 1], "3:2": [3, 2], "2:3": [2, 3]
            };

            if (!RATIOS[aspectRatio]) {
                // 如果比例不存在，返回默认值
                return {
                    x1: Math.round(videoWidth * 0.1),
                    y1: Math.round(videoHeight * 0.1),
                    x2: Math.round(videoWidth * 0.9),
                    y2: Math.round(videoHeight * 0.9)
                };
            }

            const [w, h] = RATIOS[aspectRatio];
            const targetAspect = w / h;
            const currentAspect = videoWidth / videoHeight;

            let cropWidth, cropHeight;
            if (currentAspect > targetAspect) {
                // 视频更宽，以高度为基准
                cropHeight = videoHeight;
                cropWidth = Math.round(cropHeight * targetAspect);
            } else {
                // 视频更高，以宽度为基准
                cropWidth = videoWidth;
                cropHeight = Math.round(cropWidth / targetAspect);
            }

            // 确保尺寸不超过视频尺寸
            cropWidth = Math.min(cropWidth, videoWidth);
            cropHeight = Math.min(cropHeight, videoHeight);

            // 计算居中位置并应用偏移
            const centerX = (videoWidth - cropWidth) / 2 + offsetX;
            const centerY = (videoHeight - cropHeight) / 2 + offsetY;

            // 确保在边界内
            const x1 = Math.max(0, Math.min(centerX, videoWidth - cropWidth));
            const y1 = Math.max(0, Math.min(centerY, videoHeight - cropHeight));

            return {
                x1: Math.round(x1),
                y1: Math.round(y1),
                x2: Math.round(x1 + cropWidth),
                y2: Math.round(y1 + cropHeight)
            };
        },

        draw: function(ctx, node, widgetWidth, y, widgetHeight) {
            const margin = 10;
            const canvasWidth = widgetWidth - margin * 2;

            // 获取视频尺寸
            const videoWidth = node.videoWidth || 1920;
            const videoHeight = node.videoHeight || 1080;

            // 使用自定义坐标模式：pos_x/pos_y + crop_width/crop_height
            const posX = this.getParam(node, "pos_x") ?? 0;
            const posY = this.getParam(node, "pos_y") ?? 0;
            const cropWidth = this.getParam(node, "crop_width") ?? Math.min(videoWidth * 0.8, 1920);
            const cropHeight = this.getParam(node, "crop_height") ?? Math.min(videoHeight * 0.8, 1080);

            const x1 = Math.max(0, Math.min(posX, videoWidth - cropWidth));
            const y1 = Math.max(0, Math.min(posY, videoHeight - cropHeight));
            const x2 = x1 + cropWidth;
            const y2 = y1 + cropHeight;
            const previewVideo = node.previewImagePath || "";

            // 智能计算预览区域高度 - 根据视频宽高比自适应
            const videoAspectRatio = videoWidth / videoHeight;
            const canvasAspectRatio = canvasWidth / 200; // 默认高度200作为基准

            let previewHeight;
            if (videoAspectRatio > canvasAspectRatio) {
                // 视频更宽，按宽度适配
                previewHeight = canvasWidth / videoAspectRatio;
            } else {
                // 视频更高（竖屏）或接近正方形，按可用空间适配
                const maxHeight = Math.min(400, canvasWidth / 0.5); // 最大高度限制，最小宽高比0.5
                previewHeight = Math.min(canvasWidth / videoAspectRatio, maxHeight);
            }

            // 确保最小高度
            previewHeight = Math.max(previewHeight, 150);

            // 绘制背景
            ctx.fillStyle = "#222";
            ctx.fillRect(margin, y, canvasWidth, previewHeight);

            ctx.strokeStyle = "#555";
            ctx.lineWidth = 1;
            ctx.strokeRect(margin, y, canvasWidth, previewHeight);

            // 计算缩放 - 现在预览高度是自适应的
            const scale = Math.min(canvasWidth / videoWidth, previewHeight / videoHeight);
            const scaledVideoWidth = videoWidth * scale;
            const scaledVideoHeight = videoHeight * scale;
            const offsetX = margin + (canvasWidth - scaledVideoWidth) / 2;
            const offsetY = y + (previewHeight - scaledVideoHeight) / 2;

            // 尝试显示真实视频或使用默认背景
            if (previewVideo && previewVideo.length > 0) {
                // 显示真实视频
                this.drawVideoPreview(ctx, previewVideo, offsetX, offsetY, scaledVideoWidth, scaledVideoHeight);
            } else {
                // 绘制默认视频区域和提示
                this.drawDefaultBackground(ctx, offsetX, offsetY, scaledVideoWidth, scaledVideoHeight, scale, node);
            }

            // 绘制遮罩
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(offsetX, offsetY, scaledVideoWidth, scaledVideoHeight);

            // 计算裁切框
            const cropX = offsetX + x1 * scale;
            const cropY = offsetY + y1 * scale;
            const scaledCropWidth = (x2 - x1) * scale;
            const scaledCropHeight = (y2 - y1) * scale;

            // 清除裁切区域遮罩
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillRect(cropX, cropY, scaledCropWidth, scaledCropHeight);
            ctx.globalCompositeOperation = 'source-over';

            // 绘制裁切框
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.strokeRect(cropX, cropY, scaledCropWidth, scaledCropHeight);

            // 绘制调整大小的控制点
            const handleSize = 8;
            ctx.fillStyle = '#ff4444';

            // 四个角的控制点
            ctx.fillRect(cropX - handleSize/2, cropY - handleSize/2, handleSize, handleSize);
            ctx.fillRect(cropX + scaledCropWidth - handleSize/2, cropY - handleSize/2, handleSize, handleSize);
            ctx.fillRect(cropX - handleSize/2, cropY + scaledCropHeight - handleSize/2, handleSize, handleSize);
            ctx.fillRect(cropX + scaledCropWidth - handleSize/2, cropY + scaledCropHeight - handleSize/2, handleSize, handleSize);

            // 绘制信息
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            const info = `视频: ${videoWidth}×${videoHeight} | 裁切: ${x2-x1}×${y2-y1} | 位置: (${x1}, ${y1})`;
            ctx.fillText(info, margin + 5, y + previewHeight - 8);

            // 始终保存画布信息，无论鼠标是否在上面
            this.canvasInfo = {
                offsetX, offsetY, scale, margin, y,
                canvasWidth, previewHeight, videoWidth, videoHeight,
                cropX, cropY, scaledCropWidth, scaledCropHeight
            };

            return previewHeight + 15;
        },

        getParam: function(node, name) {
            const widget = node.widgets.find(w => w.name === name);
            return widget ? widget.value : null;
        },

        setParam: function(node, name, value) {
            const widget = node.widgets.find(w => w.name === name);
            if (widget) {
                widget.value = value;
            }
        },

        mouse: function(event, pos, node) {
            console.log(`🖱️ 鼠标事件触发! 类型: ${event.type}, 位置: (${pos[0]}, ${pos[1]}), 拖拽状态: ${this.isDragging}`);

            // 🚨 优先处理 pointerup 事件，无论位置如何都要清理拖拽状态
            if (event.type === "pointerup") {
                if (this.isDragging) {
                    console.log("🖱️ 优先处理鼠标释放事件（拖拽状态） - 无视区域限制");
                    this.isDragging = false;
                    this.resizeHandle = null;
                    console.log("✅ 拖拽状态已重置");
                }
                return true; // 总是处理释放事件
            }

            // 如果没有canvasInfo，直接返回false（draw函数会在每次绘制时更新canvasInfo）
            if (!this.canvasInfo) {
                console.log("⚠️ 画布信息未准备好，等待下次绘制");
                return false;
            }

            const { offsetX, offsetY, scale, margin, y, canvasWidth, previewHeight } = this.canvasInfo;

            // 获取当前裁切坐标（与draw函数保持一致）
            const videoWidth = node.videoWidth || 1920;
            const videoHeight = node.videoHeight || 1080;
            const posX = this.getParam(node, "pos_x") ?? 0;
            const posY = this.getParam(node, "pos_y") ?? 0;
            const cropWidth = this.getParam(node, "crop_width") ?? Math.min(videoWidth * 0.8, 1920);
            const cropHeight = this.getParam(node, "crop_height") ?? Math.min(videoHeight * 0.8, 1080);

            const x1 = Math.max(0, Math.min(posX, videoWidth - cropWidth));
            const y1 = Math.max(0, Math.min(posY, videoHeight - cropHeight));
            const x2 = x1 + cropWidth;
            const y2 = y1 + cropHeight;

            console.log(`🖱️ 鼠标位置: (${pos[0]}, ${pos[1]}) | 事件类型: ${event.type}`);
            console.log(`📐 画布信息: offset(${offsetX}, ${offsetY}) scale:${scale} preview:${canvasWidth}x${previewHeight}`);

            // 对于拖拽状态下的移动事件，允许越界继续运行
            if (this.isDragging && event.type === "pointermove") {
                console.log("🔄 拖拽中的移动事件 - 允许越界");
                // 跳过区域检查，直接进入移动处理逻辑
            } else {
                // 对于非拖拽状态的事件，进行正常的区域检查
                if (pos[0] < margin || pos[0] > margin + canvasWidth ||
                    pos[1] < y || pos[1] > y + previewHeight) {
                    console.log("❌ 鼠标不在预览区域内");
                    return false;
                }

                // 计算当前缩放后的视频显示区域
                const scaledVideoWidth = videoWidth * scale;
                const scaledVideoHeight = videoHeight * scale;

                console.log(`📏 视频显示区域: (${offsetX.toFixed(1)}, ${offsetY.toFixed(1)}) 到 (${(offsetX + scaledVideoWidth).toFixed(1)}, ${(offsetY + scaledVideoHeight).toFixed(1)})`);
                console.log(`🔍 缩放后视频尺寸: ${scaledVideoWidth.toFixed(1)}×${scaledVideoHeight.toFixed(1)}`);

                if (pos[0] < offsetX || pos[0] > offsetX + scaledVideoWidth ||
                    pos[1] < offsetY || pos[1] > offsetY + scaledVideoHeight) {
                    console.log(`❌ 鼠标不在视频显示区域内: (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}) vs 区域[(${offsetX.toFixed(1)}, ${offsetY.toFixed(1)}) - (${(offsetX + scaledVideoWidth).toFixed(1)}, ${(offsetY + scaledVideoHeight).toFixed(1)})]`);
                    return false;
                }
            }

            // 计算当前缩放后的视频显示区域（为了后续计算需要）
            const scaledVideoWidth = videoWidth * scale;
            const scaledVideoHeight = videoHeight * scale;

            // 转换坐标到视频坐标系
            let videoX = (pos[0] - offsetX) / scale;
            let videoY = (pos[1] - offsetY) / scale;
            videoX = Math.max(0, Math.min(videoX, videoWidth));
            videoY = Math.max(0, Math.min(videoY, videoHeight));

            console.log(`🎯 视频坐标: (${videoX.toFixed(0)}, ${videoY.toFixed(0)})`);

            // 计算裁切框位置
            const cropX = offsetX + x1 * scale;
            const cropY = offsetY + y1 * scale;
            const scaledCropWidth = (x2 - x1) * scale;
            const scaledCropHeight = (y2 - y1) * scale;

            if (event.type === "pointerdown") {
                // 检查是否点击在控制点或红框内部
                const handleDisplaySize = 8;  // 显示大小
                const handleClickSize = 20;   // 点击区域大小（更大，更容易点击）

                // 控制点的中心位置
                const topLeftCenter = [cropX, cropY];
                const topRightCenter = [cropX + scaledCropWidth, cropY];
                const bottomLeftCenter = [cropX, cropY + scaledCropHeight];
                const bottomRightCenter = [cropX + scaledCropWidth, cropY + scaledCropHeight];

                console.log(`🎯 控制点中心: TL(${topLeftCenter[0].toFixed(0)}, ${topLeftCenter[1].toFixed(0)}) TR(${topRightCenter[0].toFixed(0)}, ${topRightCenter[1].toFixed(0)})`);

                // 重置状态
                this.resizeHandle = null;
                this.isDragging = false;
                let isValidClick = false;

                // 检查是否点击在控制点的扩大区域内（以中心点为基准）
                if (this.isPointInCircle(pos, topLeftCenter, handleClickSize/2)) {
                    this.resizeHandle = 'topLeft';
                    isValidClick = true;
                    console.log("🔍 点击左上角控制点");
                } else if (this.isPointInCircle(pos, topRightCenter, handleClickSize/2)) {
                    this.resizeHandle = 'topRight';
                    isValidClick = true;
                    console.log("🔍 点击右上角控制点");
                } else if (this.isPointInCircle(pos, bottomLeftCenter, handleClickSize/2)) {
                    this.resizeHandle = 'bottomLeft';
                    isValidClick = true;
                    console.log("🔍 点击左下角控制点");
                } else if (this.isPointInCircle(pos, bottomRightCenter, handleClickSize/2)) {
                    this.resizeHandle = 'bottomRight';
                    isValidClick = true;
                    console.log("🔍 点击右下角控制点");
                } else if (pos[0] >= cropX && pos[0] <= cropX + scaledCropWidth &&
                          pos[1] >= cropY && pos[1] <= cropY + scaledCropHeight) {
                    // 检查是否点击在裁切框内部
                    isValidClick = true;
                    console.log("🔍 点击裁切框内部区域（拖拽模式）");
                } else {
                    console.log("❌ 点击位置不在裁切框或控制点上");
                    return false;  // 不处理无效点击
                }

                // 只有有效点击才开始拖拽
                if (isValidClick) {
                    this.isDragging = true;
                    this.dragStartX = videoX;
                    this.dragStartY = videoY;
                    this.startX1 = x1;
                    this.startY1 = y1;
                    this.startX2 = x2;
                    this.startY2 = y2;
                    console.log("✅ 开始拖拽");
                    return true;
                }

                return false;
            } else if (event.type === "pointermove" && this.isDragging) {
                // 对于拖拽中的移动事件，重新计算坐标（允许越界）
                console.log("🔄 处理拖拽移动事件");

                // 直接重新计算坐标，不限制在边界内
                let actualVideoX = (pos[0] - offsetX) / scale;
                let actualVideoY = (pos[1] - offsetY) / scale;

                console.log(`🎯 计算拖拽坐标: 鼠标(${pos[0]}, ${pos[1]}) -> 视频坐标(${actualVideoX.toFixed(1)}, ${actualVideoY.toFixed(1)})`);

                const deltaX = actualVideoX - this.dragStartX;
                const deltaY = actualVideoY - this.dragStartY;

                if (this.resizeHandle) {
                    // 调整大小
                    let newX1 = this.startX1;
                    let newY1 = this.startY1;
                    let newX2 = this.startX2;
                    let newY2 = this.startY2;

                    switch (this.resizeHandle) {
                        case 'topLeft':
                            newX1 = this.startX1 + deltaX;
                            newY1 = this.startY1 + deltaY;
                            break;
                        case 'topRight':
                            newX2 = this.startX2 + deltaX;
                            newY1 = this.startY1 + deltaY;
                            break;
                        case 'bottomLeft':
                            newX1 = this.startX1 + deltaX;
                            newY2 = this.startY2 + deltaY;
                            break;
                        case 'bottomRight':
                            newX2 = this.startX2 + deltaX;
                            newY2 = this.startY2 + deltaY;
                            break;
                    }

                    // 确保裁切框不会反转
                    if (newX1 >= newX2) newX1 = newX2 - 50;
                    if (newY1 >= newY2) newY1 = newY2 - 50;

                    // 边界约束
                    newX1 = Math.max(0, Math.min(newX1, videoWidth - 50));
                    newY1 = Math.max(0, Math.min(newY1, videoHeight - 50));
                    newX2 = Math.max(50, Math.min(newX2, videoWidth));
                    newY2 = Math.max(50, Math.min(newY2, videoHeight));

                    // 更新参数
                    // 更新裁切参数
                    this.setParam(node, "pos_x", Math.round(newX1));
                    this.setParam(node, "pos_y", Math.round(newY1));
                    this.setParam(node, "crop_width", Math.round(newX2 - newX1));
                    this.setParam(node, "crop_height", Math.round(newY2 - newY1));

                    node.setDirtyCanvas(true, true);
                } else {
                    // 普通拖拽 - 保持裁切框大小不变
                    const cropWidth = this.startX2 - this.startX1;
                    const cropHeight = this.startY2 - this.startY1;

                    // 直接计算目标位置
                    let targetX1 = this.startX1 + deltaX;
                    let targetY1 = this.startY1 + deltaY;

                    // 约束到视频边界
                    targetX1 = Math.max(0, Math.min(targetX1, videoWidth - cropWidth));
                    targetY1 = Math.max(0, Math.min(targetY1, videoHeight - cropHeight));

                    // 计算对应的X2, Y2
                    const targetX2 = targetX1 + cropWidth;
                    const targetY2 = targetY1 + cropHeight;

                    // 更新参数
                    // 更新位置参数
                    this.setParam(node, "pos_x", Math.round(targetX1));
                    this.setParam(node, "pos_y", Math.round(targetY1));

                    node.setDirtyCanvas(true, true);
                }

                return true;
            }

            console.log("🖱️ 鼠标事件未处理");
            return false;
        },

        isPointInRect: function(point, rectPos, size) {
            return point[0] >= rectPos[0] && point[0] <= rectPos[0] + size &&
                   point[1] >= rectPos[1] && point[1] <= rectPos[1] + size;
        },

        isPointInCircle: function(point, center, radius) {
            const dx = point[0] - center[0];
            const dy = point[1] - center[1];
            return (dx * dx + dy * dy) <= (radius * radius);
        },

        drawDefaultBackground: function(ctx, x, y, width, height, scale, node) {
            // 绘制默认视频区域
            ctx.fillStyle = "#333";
            ctx.fillRect(x, y, width, height);

            // 绘制网格
            ctx.strokeStyle = "#666";
            ctx.lineWidth = 0.5;
            const gridSize = 50 * scale;
            for (let i = 0; i <= width; i += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x + i, y);
                ctx.lineTo(x + i, y + height);
                ctx.stroke();
            }
            for (let i = 0; i <= height; i += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, y + i);
                ctx.lineTo(x + width, y + i);
                ctx.stroke();
            }

            // 获取当前input_folder
            const inputFolder = this.getParam(node, "input_folder") ?? "input";

            // 绘制提示文本
            ctx.fillStyle = "#888";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText(`预览: "${inputFolder}" 文件夹`, x + width/2, y + height/2 - 10);
            ctx.fillText("将视频文件放入input文件夹查看预览", x + width/2, y + height/2 + 10);
        },

        drawVideoPreview: function(ctx, previewPath, x, y, width, height) {
            // 检查是否是图片路径
            if (previewPath.includes('.jpg') || previewPath.includes('.png') || previewPath.includes('.jpeg')) {
                // 处理图片预览
                this.drawImagePreview(ctx, previewPath, x, y, width, height);
                return;
            }

            // 如果已经有视频元素并且路径相同，直接绘制
            if (this.previewVideoElement && this.previewVideoPath === previewPath) {
                if (this.previewVideoElement.videoWidth > 0) {
                    ctx.drawImage(this.previewVideoElement, x, y, width, height);
                    return;
                }
            }

            // 创建新的视频元素
            if (!this.previewVideoElement || this.previewVideoPath !== previewPath) {
                this.previewVideoElement = document.createElement('video');
                this.previewVideoElement.crossOrigin = 'anonymous';
                this.previewVideoElement.muted = true;
                this.previewVideoElement.loop = true;
                this.previewVideoElement.autoplay = true;
                this.previewVideoElement.playsInline = true;
                this.previewVideoPath = previewPath;

                console.log(`🎬 加载预览视频: ${previewPath}`);

                this.previewVideoElement.onloadeddata = () => {
                    console.log(`✅ 视频加载完成，尺寸: ${this.previewVideoElement.videoWidth}×${this.previewVideoElement.videoHeight}`);
                    // 视频加载完成后重绘
                    if (this.node) this.node.setDirtyCanvas(true, true);
                };

                this.previewVideoElement.onerror = (e) => {
                    console.error(`❌ 视频加载失败:`, e);
                };

                // 设置视频源
                this.previewVideoElement.src = previewPath;
            }

            // 如果视频还在加载，显示加载提示
            if (this.previewVideoElement.videoWidth === 0) {
                ctx.fillStyle = "#444";
                ctx.fillRect(x, y, width, height);

                ctx.fillStyle = "#fff";
                ctx.font = "14px Arial";
                ctx.textAlign = "center";
                ctx.fillText("正在加载视频预览...", x + width/2, y + height/2);
            } else {
                // 绘制视频帧
                ctx.drawImage(this.previewVideoElement, x, y, width, height);
            }
        },

        drawImagePreview: function(ctx, imagePath, x, y, width, height) {
            // 如果已经有图片元素并且路径相同，直接绘制
            if (this.previewImageElement && this.previewImagePath === imagePath) {
                if (this.previewImageElement.complete) {
                    ctx.drawImage(this.previewImageElement, x, y, width, height);
                    return;
                }
            }

            // 创建新的图片元素
            if (!this.previewImageElement || this.previewImagePath !== imagePath) {
                this.previewImageElement = new Image();
                this.previewImageElement.crossOrigin = 'anonymous';
                this.previewImagePath = imagePath;

                console.log(`🖼️ 加载预览图片: ${imagePath}`);

                this.previewImageElement.onload = () => {
                    console.log(`✅ 图片加载完成，尺寸: ${this.previewImageElement.width}×${this.previewImageElement.height}`);
                    // 图片加载完成后重绘
                    if (this.node) this.node.setDirtyCanvas(true, true);
                };

                this.previewImageElement.onerror = (e) => {
                    console.error(`❌ 图片加载失败:`, e);
                };

                // 设置图片源
                this.previewImageElement.src = imagePath;
            }

            // 如果图片还在加载，显示加载提示
            if (!this.previewImageElement.complete) {
                ctx.fillStyle = "#444";
                ctx.fillRect(x, y, width, height);

                ctx.fillStyle = "#fff";
                ctx.font = "14px Arial";
                ctx.textAlign = "center";
                ctx.fillText("正在加载图片预览...", x + width/2, y + height/2);
            } else {
                // 绘制图片
                ctx.drawImage(this.previewImageElement, x, y, width, height);
            }
        },

        computeSize: function(width) {
            // 从缓存的node获取视频尺寸信息
            if (this.node) {
                const videoWidth = this.node.videoWidth || 1920;
                const videoHeight = this.node.videoHeight || 1080;

                const margin = 10;
                const canvasWidth = width - margin * 2;
                const videoAspectRatio = videoWidth / videoHeight;
                const canvasAspectRatio = canvasWidth / 200; // 默认高度200作为基准

                let previewHeight;
                if (videoAspectRatio > canvasAspectRatio) {
                    // 视频更宽，按宽度适配
                    previewHeight = canvasWidth / videoAspectRatio;
                } else {
                    // 视频更高（竖屏）或接近正方形，按可用空间适配
                    const maxHeight = Math.min(400, canvasWidth / 0.5); // 最大高度限制，最小宽高比0.5
                    previewHeight = Math.min(canvasWidth / videoAspectRatio, maxHeight);
                }

                // 确保最小高度
                previewHeight = Math.max(previewHeight, 150);

                // 返回总高度（预览区域 + 底部信息区域）
                return [width, previewHeight + 15];
            }

            // 默认尺寸（如果没有node引用）
            return [width, 220];
        }
    };
}

// 注册扩展
app.registerExtension({
    name: "VideoEditing.SimpleCrop",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EnhancedVideoCropNode") {
            console.log("✅ 找到EnhancedVideoCropNode节点！");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                console.log("📋 创建视频裁切节点...");

                // 添加比例选择器
                const ratioWidget = createRatioSelector(this);
                this.addCustomWidget(ratioWidget);

                // 添加预览界面
                const previewWidget = createPreview(this);
                previewWidget.node = this; // 保存node引用以便动态计算尺寸
                this.addCustomWidget(previewWidget);

                console.log("🎯 Widget信息:", {
                    ratioWidget: ratioWidget.name,
                    previewWidget: previewWidget.name,
                    totalWidgets: this.widgets.length,
                    hasMouse: !!previewWidget.mouse
                });

                // 延迟尝试加载预览视频，确保所有widgets都已初始化（静默模式避免初始404错误）
                setTimeout(() => {
                    this.tryLoadPreviewVideo(0, true);
                }, 100);

                // 监听input_folder变化
                this.setupInputFolderListener();

                // 添加节点级别的鼠标事件监听
                this.setupNodeMouseHandler(previewWidget);

                // 添加全局鼠标释放事件监听
                this.setupGlobalMouseHandler(previewWidget);

                console.log("✅ 视频裁切界面已添加");

                return result;
            };

            // 简化的预览图片加载函数（静默模式，不产生错误日志）
            nodeType.prototype.tryLoadPreviewVideo = async function(retryCount = 0, silent = true) {
                if (!silent) {
                    console.log(`🎬 尝试加载预览图片... (重试次数: ${retryCount})`);
                }

                // 获取当前输入文件夹
                const inputFolderWidget = this.widgets.find(w => w.name === "input_folder");
                if (!inputFolderWidget) {
                    if (!silent) {
                        console.log("⚠️ 未找到input_folder参数");
                    }
                    // 如果重试次数少于3次，延迟重试
                    if (retryCount < 3) {
                        setTimeout(() => {
                            this.tryLoadPreviewVideo(retryCount + 1, silent);
                        }, 500);
                    }
                    return;
                }

                const inputFolder = inputFolderWidget.value || "input";
                if (!silent) {
                    console.log(`📁 输入文件夹: ${inputFolder}`);
                }

                // 计算输入目录的路径哈希，与Python端保持一致
                const inputFolderPath = `/home/shenglin/Desktop/ComfyUI/input/${inputFolder}`;
                const inputPathHash = await calculateMD5Hash(inputFolderPath);

                // 生成预览图片的路径 - 通过ComfyUI的output view端点访问，包含路径哈希
                const previewImagePath = `/view?filename=video_preview_${inputFolder}_${inputPathHash}.jpg&type=output`;

                if (!silent) {
                    console.log(`🔍 尝试访问预览图片: ${previewImagePath}`);
                }

                // 测试预览图片是否存在
                const testImage = new Image();
                testImage.crossOrigin = 'anonymous';

                testImage.onload = () => {
                    if (!silent) {
                        console.log(`✅ 找到预览图片: ${previewImagePath}`);
                        console.log(`📏 图片尺寸: ${testImage.width}×${testImage.height}`);
                    }

                    // 保存视频尺寸和预览图片路径到节点属性（内部使用）
                    if (testImage.width > 0) {
                        this.videoWidth = testImage.width;
                        this.videoHeight = testImage.height;
                        this.previewImagePath = previewImagePath;
                        if (!silent) {
                            console.log(`📏 已保存视频尺寸: ${testImage.width}×${testImage.height}`);
                        }

                        // 视频尺寸变化时重新计算widget尺寸
                        const previewWidget = this.widgets.find(w => w.type === "crop_preview");
                        if (previewWidget) {
                            // 触发尺寸重新计算
                            this.computeSize();
                            this.setSize(this.size);
                        }
                    }

                    // 强制重绘
                    this.setDirtyCanvas(true, true);
                };

                testImage.onerror = (e) => {
                    if (!silent) {
                        console.log(`❌ 无法访问预览图片: ${previewImagePath}`);
                    }

                    // 自动生成预览图片（除非是静默的初始化加载）
                    if (!silent) {
                        console.log("🔄 预览图片不存在，开始自动生成...");
                        this.generatePreviewImage(inputFolder);
                    }
                };

                // 开始加载图片
                testImage.src = previewImagePath;
            };

            // 生成预览图片的函数 - 切换文件夹时自动生成预览
            nodeType.prototype.generatePreviewImage = function(inputFolder) {
                console.log(`🎨 开始为文件夹 "${inputFolder}" 生成预览图片...`);

                // 创建一个简单的工作流来生成预览图片（使用最小裁切尺寸以确保兼容性）
                const previewWorkflow = {
                    "1": {
                        "inputs": {
                            "input_folder": inputFolder,
                            "output_folder_name": "auto_preview",
                            "aspect_ratio": "自定义",
                            "pos_x": 0,
                            "pos_y": 0,
                            "crop_width": 320,
                            "crop_height": 240
                        },
                        "class_type": "EnhancedVideoCropNode"
                    },
                    "2": {
                        "inputs": {
                            "text": ["1", 0]
                        },
                        "class_type": "ShowText|pysssss",
                        "_meta": {
                            "title": "Preview Output"
                        }
                    }
                };

                // 发送工作流到ComfyUI执行
                fetch('/prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: previewWorkflow,
                        client_id: Math.random().toString(36).substr(2, 9)
                    })
                })
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    }
                    // 获取详细错误信息
                    return response.text().then(text => {
                        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${text}`);
                    });
                })
                .then(data => {
                    console.log("✅ 预览图片生成请求已提交", data);

                    // 等待3秒后重新尝试加载预览
                    setTimeout(() => {
                        this.tryLoadPreviewVideo(0, false);
                    }, 3000);
                })
                .catch(error => {
                    console.log("❌ 预览图片生成失败:", error.message);
                    console.log("🔧 工作流内容:", JSON.stringify(previewWorkflow, null, 2));
                });
            };

            // 监听input_folder变化
            nodeType.prototype.setupInputFolderListener = function() {
                console.log("👂 设置input_folder变化监听...");

                const inputFolderWidget = this.widgets.find(w => w.name === "input_folder");
                if (!inputFolderWidget) {
                    console.log("⚠️ 未找到input_folder控件");
                    return;
                }

                // 保存原始callback
                const originalCallback = inputFolderWidget.callback;

                // 添加自定义callback
                inputFolderWidget.callback = (value) => {
                    console.log(`📁 input_folder已更改为: ${value}`);

                    // 执行原始callback
                    if (originalCallback) {
                        originalCallback.call(inputFolderWidget, value);
                    }

                    // 清除当前预览
                    this.clearPreview();

                    // 防抖：取消之前的定时器
                    if (this.previewTimer) {
                        clearTimeout(this.previewTimer);
                    }

                    // 设置新的定时器（非静默模式，用户主动切换文件夹）
                    this.previewTimer = setTimeout(() => {
                        this.tryLoadPreviewVideo(0, false);
                    }, 500); // 500ms防抖延迟
                };

                console.log("✅ input_folder监听已设置");
            };

            // 清除预览函数
            nodeType.prototype.clearPreview = function() {
                console.log("🧹 清除当前预览...");

                // 清除节点属性
                this.previewImagePath = "";
                this.videoWidth = 1920;
                this.videoHeight = 1080;

                // 清除缓存的视频元素
                const previewWidget = this.widgets.find(w => w.type === "crop_preview");
                if (previewWidget) {
                    previewWidget.previewVideoElement = null;
                    previewWidget.previewVideoPath = "";
                }

                // 强制重绘
                this.setDirtyCanvas(true, true);
            };

            // 添加节点级别的鼠标事件处理
            nodeType.prototype.setupNodeMouseHandler = function(previewWidget) {
                console.log("🖱️ 设置节点级别鼠标事件监听...");

                // 覆盖节点的onMouseDown方法
                const originalOnMouseDown = this.onMouseDown;
                this.onMouseDown = function(e, localPos, graphCanvas) {
                    console.log(`🖱️ 节点鼠标按下事件: (${localPos[0]}, ${localPos[1]})`);

                    // 尝试将事件传递给预览widget
                    if (previewWidget && previewWidget.mouse) {
                        const handled = previewWidget.mouse({type: "pointerdown"}, localPos, this);
                        if (handled) {
                            console.log("✅ 预览widget处理了鼠标事件");
                            return true;
                        }
                    }

                    // 如果预览widget没有处理，调用原始方法
                    if (originalOnMouseDown) {
                        return originalOnMouseDown.call(this, e, localPos, graphCanvas);
                    }
                    return false;
                };

                // 覆盖节点的onMouseMove方法
                const originalOnMouseMove = this.onMouseMove;
                this.onMouseMove = function(e, localPos, graphCanvas) {
                    // 只有在拖拽时才处理鼠标移动事件
                    if (previewWidget && previewWidget.isDragging) {
                        console.log(`🖱️ 节点级别处理拖拽移动: (${localPos[0]}, ${localPos[1]})`);
                        if (previewWidget.mouse) {
                            const handled = previewWidget.mouse({type: "pointermove"}, localPos, this);
                            if (handled) {
                                return true;
                            }
                        }
                    }

                    // 对于非拖拽的鼠标移动，调用原始方法但不重绘
                    if (originalOnMouseMove) {
                        return originalOnMouseMove.call(this, e, localPos, graphCanvas);
                    }
                    return false;
                };

                // 覆盖节点的onMouseUp方法
                const originalOnMouseUp = this.onMouseUp;
                this.onMouseUp = function(e, localPos, graphCanvas) {
                    // 无论鼠标在哪里释放，如果预览widget正在拖拽，都要处理释放事件
                    if (previewWidget && previewWidget.isDragging) {
                        console.log("🖱️ 强制处理鼠标释放事件（拖拽状态）");
                        previewWidget.isDragging = false;
                        previewWidget.resizeHandle = null;
                        console.log("✅ 拖拽状态已重置");
                        return true;
                    }

                    if (originalOnMouseUp) {
                        return originalOnMouseUp.call(this, e, localPos, graphCanvas);
                    }
                    return false;
                };

                console.log("✅ 节点级别鼠标事件监听已设置");
            };

            // 添加全局鼠标事件处理，确保拖拽状态能被正确重置
            nodeType.prototype.setupGlobalMouseHandler = function(previewWidget) {
                console.log("🌐 设置全局鼠标事件监听...");

                // 添加全局鼠标释放事件监听
                const globalMouseUp = (e) => {
                    if (previewWidget && previewWidget.isDragging) {
                        console.log("🌐 全局鼠标释放事件 - 重置拖拽状态");
                        previewWidget.isDragging = false;
                        previewWidget.resizeHandle = null;
                    }
                };

                // 绑定到document
                document.addEventListener('mouseup', globalMouseUp);
                document.addEventListener('pointerup', globalMouseUp);

                // 保存引用以便清理
                this._globalMouseUpHandler = globalMouseUp;

                console.log("✅ 全局鼠标事件监听已设置");
            };

            // 清理全局事件监听
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log("🧹 清理全局事件监听...");

                if (this._globalMouseUpHandler) {
                    document.removeEventListener('mouseup', this._globalMouseUpHandler);
                    document.removeEventListener('pointerup', this._globalMouseUpHandler);
                    this._globalMouseUpHandler = null;
                }

                if (originalOnRemoved) {
                    originalOnRemoved.call(this);
                }
            };
        }
    }
});

console.log("🎬 视频裁切扩展已注册 - 简化版本");