/**
 * 视频预览节点的前端支持
 * 在ComfyUI界面中显示HTML5视频播放器
 */

import { app } from "../../web/scripts/app.js";

// 创建视频元素的函数
function createVideoElement(src, options = {}) {
    const video = document.createElement('video');
    video.src = src;
    video.controls = options.controls !== false;
    video.autoplay = options.autoplay === true;
    video.loop = options.loop === true;
    video.muted = options.autoplay === true; // 现代浏览器要求自动播放视频静音

    video.style.cssText = `
        width: 100%;
        max-width: 512px;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    `;

    return video;
}

// 注册自定义节点类型
app.registerExtension({
    name: "VideoPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VideoPreview") {
            // 添加视频显示功能
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                if (message?.video) {
                    this.displayVideo(message.video);
                }
            };

            // 显示视频的方法
            nodeType.prototype.displayVideo = function(videoData) {
                // 清除之前的视频
                if (this.videoContainer) {
                    this.videoContainer.remove();
                    this.videoContainer = null;
                }

                if (!videoData || videoData.length === 0) {
                    return;
                }

                const videoInfo = videoData[0];

                // 检查是否有错误
                if (videoInfo.error) {
                    this.displayError(videoInfo.error);
                    return;
                }

                // 创建视频容器
                this.videoContainer = document.createElement('div');
                this.videoContainer.style.cssText = `
                    position: absolute;
                    background: rgba(0,0,0,0.8);
                    border-radius: 8px;
                    padding: 10px;
                    margin-top: 10px;
                    z-index: 1000;
                `;

                // 创建视频元素
                const video = createVideoElement(videoInfo.path, {
                    controls: videoInfo.controls,
                    autoplay: videoInfo.autoplay,
                    loop: videoInfo.loop
                });

                // 添加视频信息
                const infoDiv = document.createElement('div');
                infoDiv.style.cssText = `
                    color: white;
                    font-size: 12px;
                    margin-bottom: 8px;
                    text-align: center;
                `;

                let infoText = `📁 ${videoInfo.filename}`;
                if (videoInfo.size_mb) {
                    infoText += ` (${videoInfo.size_mb} MB)`;
                }
                if (videoInfo.width && videoInfo.height) {
                    infoText += ` | ${videoInfo.width}×${videoInfo.height}`;
                }
                if (videoInfo.duration) {
                    infoText += ` | ${Math.round(videoInfo.duration)}秒`;
                }

                infoDiv.textContent = infoText;

                this.videoContainer.appendChild(infoDiv);
                this.videoContainer.appendChild(video);

                // 添加到DOM
                document.body.appendChild(this.videoContainer);

                // 定位到节点位置
                this.updateVideoPosition();

                // 监听视频加载事件
                video.addEventListener('loadedmetadata', () => {
                    console.log(`🎥 视频加载完成: ${videoInfo.filename}`);
                });

                video.addEventListener('error', (e) => {
                    console.error('视频播放错误:', e);
                    this.displayError('视频播放失败，请检查文件格式');
                });
            };

            // 显示错误信息
            nodeType.prototype.displayError = function(errorMsg) {
                if (this.videoContainer) {
                    this.videoContainer.remove();
                }

                this.videoContainer = document.createElement('div');
                this.videoContainer.style.cssText = `
                    position: absolute;
                    background: rgba(200,0,0,0.8);
                    color: white;
                    border-radius: 8px;
                    padding: 10px;
                    margin-top: 10px;
                    max-width: 300px;
                    font-size: 12px;
                    z-index: 1000;
                `;

                this.videoContainer.textContent = `❌ ${errorMsg}`;
                document.body.appendChild(this.videoContainer);
                this.updateVideoPosition();
            };

            // 更新视频位置
            nodeType.prototype.updateVideoPosition = function() {
                if (!this.videoContainer) return;

                const rect = this.getBounding();
                const canvasRect = app.canvas.getBoundingClientRect();
                const transform = app.canvas.ds;

                // 计算节点在屏幕上的位置
                const x = (rect[0] * transform.scale) + transform.offset[0] + canvasRect.left;
                const y = (rect[1] * transform.scale) + transform.offset[1] + canvasRect.top + rect[3] * transform.scale;

                this.videoContainer.style.left = `${x}px`;
                this.videoContainer.style.top = `${y}px`;
            };

            // 清理函数
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this.videoContainer) {
                    this.videoContainer.remove();
                    this.videoContainer = null;
                }
                onRemoved?.apply(this, arguments);
            };

            // 画布变化时更新位置
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                const r = onDrawBackground?.apply(this, arguments);

                // 延迟更新位置，避免频繁调用
                if (this.videoContainer && !this.positionUpdatePending) {
                    this.positionUpdatePending = true;
                    requestAnimationFrame(() => {
                        this.updateVideoPosition();
                        this.positionUpdatePending = false;
                    });
                }

                return r;
            };
        }
    }
});

console.log("🎥 视频预览扩展已加载");