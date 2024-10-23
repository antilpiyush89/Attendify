// Call the run function to load models initially

const loadModels = async () => {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.tinyYolov2.loadFromUri('/models'),
        faceapi.nets.mtcnn.loadFromUri('/models')
    ]);
}

loadModels()
    .then(() => {
        console.log("Models loaded successfully.");
        // Now you can perform inference, e.g., start video feed or process images.
        let labeledFaceDescriptors;  // Store the labeled face descriptor

    // Handle image upload and extract face descriptor
    const handleImageUpload = async (event) => {
        const imageFile = event.target.files[0];
        const img = await faceapi.bufferToImage(imageFile);

        const uploadedFace = await faceapi.detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();

        if (!uploadedFace) {
            console.log("No face detected in the uploaded image.");
            return;
        }

        const uploadedFaceDescriptor = uploadedFace.descriptor;
        
        // Create labeled face descriptors from the uploaded image
        labeledFaceDescriptors = [new faceapi.LabeledFaceDescriptors('Uploaded Face', [uploadedFaceDescriptor])];
        
        console.log("Face successfully uploaded and processed.");

        // Now, open the webcam to start detecting faces
        startWebcam();
    };

    // Function to start the webcam and match the face
    const startWebcam = () => {
        console.log("Starting webcam...");
        const videoEl = document.getElementById('video-feed');
        navigator.getUserMedia(
            { video: {} },
            stream => videoEl.srcObject = stream,
            err => console.error(err)
        );

        videoEl.addEventListener('play', async () => {
            console.log("Webcam started playing.");
            const canvas = faceapi.createCanvasFromMedia(videoEl);
            document.body.append(canvas);
            const displaySize = { width: videoEl.width, height: videoEl.height };
            faceapi.matchDimensions(canvas, displaySize);
            console.log("Canvas dimensions matched.");
            const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6); // Increased threshold to 0.6
            console.log("Face matcher created.");

            setInterval(async () => {
                // Detect faces in the webcam stream
                const detections = await faceapi.detectAllFaces(videoEl, new faceapi.TinyFaceDetectorOptions())
                    .withFaceLandmarks()
                    .withFaceDescriptors();
                console.log("Faces detected:", detections.length);
                const resizedDetections = faceapi.resizeResults(detections, displaySize);
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                faceapi.draw.drawDetections(canvas, resizedDetections);
                faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

                const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

                results.forEach((result, i) => {
                    const box = resizedDetections[i].detection.box;
                    const score = result.distance;
                    let text = result.toString();
                    if (score <= 0.6) {
                        text = `Match: ${(1 - score).toFixed(2)}`;
                    } else {
                        text = "No Match";
                    }
                    console.log(`Face ${i + 1}: ${text}`);

                    const drawBox = new faceapi.draw.DrawBox(box, { label: text });
                    drawBox.draw(canvas);
                });
            }, 100);
        });
    };

        // Add event listener for image upload
        document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
    })
    .catch(err => {
        console.error("Error loading models:", err);
    });
