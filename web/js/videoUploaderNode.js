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
    name: "RunningHub.VideoUploader",

    nodeCreated(node) {
        if (node.comfyClass !== "RH_VideoUploader") {
            return;
        }

        // Find the regular STRING widget that will hold the ComfyUI filename
        const filenameWidget = node.widgets.find((w) => w.name === "video");
        if (!filenameWidget) {
            console.error("RH_VideoUploader: Could not find 'video' widget on node:", node.id);
            // Might need adjustment if widget name differs or isn't created by default
            return;
        }
        // Make the default widget visually hidden initially if desired, or style it smaller.
        // filenameWidget.inputEl.style.display = 'none'; // Example: hide the input box

        // --- Create Custom UI Elements (Button, Preview, Status) ---
        const container = document.createElement("div");
        container.style.margin = "5px 0";

        const uploadButton = createElement("button", {
            textContent: "Select Video", // Changed text
            style: "width: 100%; margin-bottom: 5px;"
        });

        const previewArea = createElement("div", {
            style: "width: 100%; max-width: 256px; aspect-ratio: 16/9; background: #333; border-radius: 4px; margin-bottom: 5px; overflow: hidden; display: none; position: relative; margin-left: auto; margin-right: auto;"
        });
        const previewVideo = createElement("video", {
            controls: true,
            style: "width: 100%; height: 100%; display: block;"
        });
        const statusText = createElement("p", {
            textContent: "No video selected.", // Updated initial text
            style: "margin: 0; padding: 2px 5px; font-size: 0.8em; color: #ccc; text-align: center;"
        });

        previewArea.appendChild(previewVideo);
        container.appendChild(uploadButton);
        container.appendChild(previewArea);
        container.appendChild(statusText);

        // Add the custom UI below the standard widgets
        node.addDOMWidget("video_uploader_widget", "preview", container);

        // Function to upload file to ComfyUI backend
        async function uploadFileToComfyUI(file) {
            statusText.textContent = `Uploading ${file.name} to ComfyUI...`;
            uploadButton.disabled = true;
            uploadButton.textContent = "Uploading...";
            filenameWidget.value = ""; // Clear previous filename

            try {
                // Create FormData object
                const body = new FormData();
                // IMPORTANT: Use 'image' as the key, ComfyUI's endpoint expects this even for videos
                body.append('image', file);
                // Set overwrite=true if you want to replace existing files with the same name
                // body.append('overwrite', 'true');

                // Use ComfyUI's api.fetchApi for the upload
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                });

                if (resp.status === 200 || resp.status === 201) {
                    const data = await resp.json();
                    // data should contain { name: string, subfolder: string, type: string (input/temp) }
                    if (data.name) {
                        // We typically only need the filename, maybe prepend subfolder if needed by Python logic
                        const comfyFilename = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                        filenameWidget.value = comfyFilename; // Set the Python node's input widget
                        statusText.textContent = `Video ready: ${comfyFilename}`;
                        uploadButton.textContent = "Video Selected";
                        console.log(`RH_VideoUploader: ComfyUI upload successful: ${comfyFilename}`);
                    } else {
                        throw new Error("Filename not found in ComfyUI upload response.");
                    }
                } else {
                    throw new Error(`ComfyUI upload failed: ${resp.status} ${resp.statusText}`);
                }
            } catch (error) {
                console.error("RH_VideoUploader: ComfyUI upload error:", error);
                statusText.textContent = `Error uploading to ComfyUI: ${error.message}`;
                filenameWidget.value = "ERROR"; // Indicate error in the widget
            } finally {
                 uploadButton.disabled = false; // Re-enable button
                 // Keep text as "Video Selected" or revert?
                 if (filenameWidget.value && filenameWidget.value !== "ERROR") {
                     uploadButton.textContent = "Select Another Video";
                 } else {
                     uploadButton.textContent = "Select Video";
                 }
            }
        }

        // --- Event Listener for Button --- 
        uploadButton.addEventListener("click", () => {
            const fileInput = createElement("input", {
                type: "file",
                accept: "video/mp4,video/webm,video/ogg,video/quicktime,video/x-matroska,video/*",
                style: "display: none;"
            });

            fileInput.addEventListener("change", (event) => {
                const file = event.target.files[0];
                if (file) {
                    // Show preview
                    try {
                        const objectURL = URL.createObjectURL(file);
                        previewVideo.src = objectURL;
                        if (previewVideo.dataset.objectUrl) {
                            URL.revokeObjectURL(previewVideo.dataset.objectUrl);
                        }
                        previewVideo.dataset.objectUrl = objectURL;
                        previewArea.style.display = "block";
                    } catch (e) {
                        console.error("Error creating object URL for preview:", e);
                        previewArea.style.display = "none";
                        statusText.textContent = "Preview failed. Ready to upload."; // Update status if preview fails
                    }

                    // Upload the file to ComfyUI backend
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
            if (previewVideo.dataset.objectUrl) {
                URL.revokeObjectURL(previewVideo.dataset.objectUrl);
            }
            onRemoved?.();
        };
    },
}); 