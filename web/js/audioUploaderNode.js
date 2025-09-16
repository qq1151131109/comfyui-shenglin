import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js';

// Basic helper for creating elements
function createElement(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    Object.assign(el, attrs);
    children.forEach(child => el.appendChild(typeof child === 'string' ? document.createTextNode(child) : child));
    return el;
}

app.registerExtension({
    name: "RunningHub.AudioUploader", // Changed name

    nodeCreated(node) {
        // Match the new Python class name mapping key
        if (node.comfyClass !== "RH_AudioUploader") {
            return;
        }

        // Find the regular STRING widget that will hold the ComfyUI filename
        const filenameWidget = node.widgets.find((w) => w.name === "audio"); // Changed widget name
        if (!filenameWidget) {
            console.error("RH_AudioUploader: Could not find 'audio' widget on node:", node.id);
            return;
        }

        // --- Create Custom UI Elements ---
        const container = document.createElement("div");
        container.style.margin = "5px 0";

        const uploadButton = createElement("button", {
            textContent: "Select Audio", // Changed text
            style: "width: 100%; margin-bottom: 5px;"
        });

        // Use <audio> element for preview
        const previewArea = createElement("div", {
             // Basic styling for the container, adjust as needed
            style: "width: 100%; background: #333; border-radius: 4px; margin-bottom: 5px; overflow: hidden; display: none; padding: 5px;"
        });
        const previewAudio = createElement("audio", {
            controls: true, // Show browser default controls
            style: "width: 100%; display: block;"
        });
        const statusText = createElement("p", {
            textContent: "No audio selected.", // Updated initial text
            style: "margin: 5px 0 0 0; padding: 2px 5px; font-size: 0.8em; color: #ccc; text-align: center;"
        });

        previewArea.appendChild(previewAudio);
        container.appendChild(uploadButton);
        container.appendChild(previewArea);
        container.appendChild(statusText);

        node.addDOMWidget("audio_uploader_widget", "preview", container);

        // Function to upload file to ComfyUI backend
        async function uploadFileToComfyUI(file) {
            statusText.textContent = `Uploading ${file.name} to ComfyUI...`;
            uploadButton.disabled = true;
            uploadButton.textContent = "Uploading...";
            filenameWidget.value = "";

            try {
                const body = new FormData();
                body.append('image', file); // Still use 'image' key for ComfyUI endpoint
                
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                });

                if (resp.status === 200 || resp.status === 201) {
                    const data = await resp.json();
                    if (data.name) {
                        const comfyFilename = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                        filenameWidget.value = comfyFilename;
                        statusText.textContent = `Audio ready: ${comfyFilename}`;
                        uploadButton.textContent = "Audio Selected";
                        console.log(`RH_AudioUploader: ComfyUI upload successful: ${comfyFilename}`);
                    } else {
                        throw new Error("Filename not found in ComfyUI upload response.");
                    }
                } else {
                    throw new Error(`ComfyUI upload failed: ${resp.status} ${resp.statusText}`);
                }
            } catch (error) {
                console.error("RH_AudioUploader: ComfyUI upload error:", error);
                statusText.textContent = `Error uploading to ComfyUI: ${error.message}`;
                filenameWidget.value = "ERROR";
            } finally {
                 uploadButton.disabled = false;
                 if (filenameWidget.value && filenameWidget.value !== "ERROR") {
                     uploadButton.textContent = "Select Another Audio";
                 } else {
                     uploadButton.textContent = "Select Audio";
                 }
            }
        }

        // --- Event Listener for Button --- 
        uploadButton.addEventListener("click", () => {
            const fileInput = createElement("input", {
                type: "file",
                // Update accepted file types for audio
                accept: "audio/mpeg,audio/ogg,audio/wav,audio/aac,audio/flac,audio/*", 
                style: "display: none;"
            });

            fileInput.addEventListener("change", (event) => {
                const file = event.target.files[0];
                if (file) {
                    // Show preview using <audio> element
                    try {
                        const objectURL = URL.createObjectURL(file);
                        previewAudio.src = objectURL;
                        if (previewAudio.dataset.objectUrl) {
                            URL.revokeObjectURL(previewAudio.dataset.objectUrl);
                        }
                        previewAudio.dataset.objectUrl = objectURL;
                        previewArea.style.display = "block";
                    } catch (e) {
                        console.error("Error creating object URL for audio preview:", e);
                        previewArea.style.display = "none";
                        statusText.textContent = "Preview failed. Ready to upload.";
                    }

                    uploadFileToComfyUI(file);
                }
                fileInput.remove();
            });

            document.body.appendChild(fileInput);
            fileInput.click();
        });

        // Clean up object URL when node is removed
        const onRemoved = node.onRemoved;
        node.onRemoved = () => {
            if (previewAudio.dataset.objectUrl) {
                URL.revokeObjectURL(previewAudio.dataset.objectUrl);
            }
            onRemoved?.();
        };
    },
}); 