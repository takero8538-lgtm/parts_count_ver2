if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(reg => console.log('Service Worker 登録成功:', reg))
      .catch(err => console.error('Service Worker 登録失敗:', err));
  });
}

const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const video = document.getElementById('video');

const colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan'];

let models = {};       // { modelName: tf.GraphModel }
let imgElement = null;
let stream = null;

// モデルリストを読み込み＆複数モデルの読み込み
async function loadModels() {
  resultDiv.textContent = 'モデルを読み込み中...';
  runBtn.disabled = true;

  try {
    const res = await fetch('models_list.json');
    const modelList = await res.json();
    models = {};
    let readCount = 0;

    for (const m of modelList) {
      try {
        const model = await tf.loadGraphModel(m.path + 'model.json');
        models[m.name] = model;
        readCount++;
        console.log(`${m.name} 読み込み成功`);
      } catch (e) {
        console.warn(`${m.name} 読み込み失敗: ${e.message}`);
      }
    }

    if (readCount === 0) {
      resultDiv.textContent = 'モデルが１つも読み込めませんでした。';
    } else {
      resultDiv.textContent = `モデル ${readCount} 件読み込み完了。画像を選択または撮影してください。`;
    }
  } catch (error) {
    resultDiv.textContent = 'モデル一覧の読み込みに失敗しました。';
    console.error(error);
  }

  runBtn.disabled = !(imgElement && Object.keys(models).length > 0);
}

// 画像ファイル読み込み
imageInput.addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (!file) return;

  stopCamera();

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      imgElement = img;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = !(Object.keys(models).length > 0 && imgElement);
      resultDiv.textContent = '';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

// カメラ起動
startCameraBtn.addEventListener('click', async () => {
  if (stream) {
    stopCamera();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });
    video.srcObject = stream;
    video.style.display = 'block';
    startCameraBtn.textContent = '❌ カメラを閉じる';
    captureBtn.disabled = false;
    resultDiv.textContent = 'カメラが起動しました。対象を映して「写真を撮る」を押してください。';
  } catch (error) {
    console.error('カメラ起動エラー:', error);
    resultDiv.textContent = 'カメラの起動に失敗しました。アクセス権限を確認してください。';
  }
});

// カメラ停止
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.style.display = 'none';
  startCameraBtn.textContent = '📸 カメラを起動';
  captureBtn.disabled = true;
}

// 写真撮影
captureBtn.addEventListener('click', () => {
  if (!stream) return;

  const vW = video.videoWidth;
  const vH = video.videoHeight;
  canvas.width = vW;
  canvas.height = vH;
  ctx.drawImage(video, 0, 0, vW, vH);

  const img = new Image();
  img.src = canvas.toDataURL('image/jpeg');
  img.onload = () => {
    imgElement = img;
    runBtn.disabled = !(Object.keys(models).length > 0 && imgElement);
    resultDiv.textContent = '写真を撮影しました。「推論開始」を押してください。';
    stopCamera();
  };
});

// 単一モデルに対して推論＋NMS処理し描画
async function runInferenceWithModel(model, img, color) {
  const modelWidth = 640;
  const modelHeight = 640;
  const origWidth = img.width;
  const origHeight = img.height;

  const scale = Math.min(modelWidth / origWidth, modelHeight / origHeight);
  const nw = Math.floor(origWidth * scale);
  const nh = Math.floor(origHeight * scale);

  let inputTensor = tf.browser.fromPixels(img).toFloat();
  let resized = tf.image.resizeBilinear(inputTensor, [nh, nw]);

  const padTop = Math.floor((modelHeight - nh) / 2);
  const padLeft = Math.floor((modelWidth - nw) / 2);
  let padded = resized.pad([[padTop, modelHeight - nh - padTop], [padLeft, modelWidth - nw - padLeft], [0, 0]]);

  let expanded = padded.expandDims(0);
  let normalized = expanded.div(255.0);

  try {
    const outputTensor = await model.executeAsync(normalized);

    let rawOutput;
    if (Array.isArray(outputTensor)) {
      rawOutput = outputTensor[0];
      outputTensor.forEach(t => { if(t !== rawOutput) t.dispose(); });
    } else {
      rawOutput = outputTensor;
    }

    const squeezed = rawOutput.squeeze();
    const transposed = squeezed.transpose([1, 0]);
    const data = await transposed.data();
    const shape = transposed.shape;

    const numBoxes = shape[0];
    const numAttributes = shape[1];
    const numClasses = numAttributes - 4;

    const boxes = [];
    const scores = [];
    const classIds = [];

    const confThreshold = 0.1;

    for (let i = 0; i < numBoxes; i++) {
      const offset = i * numAttributes;

      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      let maxScore = 0;
      let classId = -1;
      for (let c = 0; c < numClasses; c++) {
        const score = data[offset + 4 + c];
        if (score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      if (maxScore >= confThreshold) {
        const ymin = cy - h / 2;
        const xmin = cx - w / 2;
        const ymax = cy + h / 2;
        const xmax = cx + w / 2;

        boxes.push([ymin, xmin, ymax, xmax]);
        scores.push(maxScore);
        classIds.push(classId);
      }
    }

    // NMS用にbbox形式変換 [xmin, ymin, width, height]
    const boxesForNMS = boxes.map(([ymin, xmin, ymax, xmax]) => {
      return [xmin, ymin, xmax - xmin, ymax - ymin];
    });

    const boxesTensor = tf.tensor2d(boxesForNMS);
    const scoresTensor = tf.tensor1d(scores);

    const maxOutputSize = 100;
    const iouThreshold = 0.45;

    const selectedIndices = await tf.image.nonMaxSuppressionAsync(
      boxesTensor,
      scoresTensor,
      maxOutputSize,
      iouThreshold,
      confThreshold
    );

    const indices = await selectedIndices.data();

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.font = '16px Arial';
    ctx.fillStyle = color;

    const results = [];

    for (const i of indices) {
      const [ymin, xmin, ymax, xmax] = boxes[i];
      const score = scores[i];
      const classId = classIds[i];

      const realXmin = (xmin - padLeft) / scale;
      const realYmin = (ymin - padTop) / scale;
      const realXmax = (xmax - padLeft) / scale;
      const realYmax = (ymax - padTop) / scale;

      const boxWidth = realXmax - realXmin;
      const boxHeight = realYmax - realYmin;

      if (boxWidth > 0 && boxHeight > 0) {
        ctx.strokeRect(realXmin, realYmin, boxWidth, boxHeight);
        // 必要ならラベルも描画可能
        // ctx.fillText(`${classId} ${(score*100).toFixed(1)}%`, realXmin + 5, realYmin + 18);
      }

      results.push({ classId, score });
    }

    boxesTensor.dispose();
    scoresTensor.dispose();
    selectedIndices.dispose();
    squeezed.dispose();
    transposed.dispose();
    rawOutput.dispose();
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);

    return results.length;

  } catch (error) {
    console.error('推論エラー', error);
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);
    return 0;
  }
}

// 複数モデル一括推論のメイン処理
async function runInferenceAllModels() {
  if (!imgElement || Object.keys(models).length === 0) {
    alert('画像またはモデルがありません。');
    return;
  }

  canvas.width = imgElement.width;
  canvas.height = imgElement.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgElement, 0, 0);

  let idx = 0;
  let totalDetections = 0;
  let outputText = '';

  for (const [name, model] of Object.entries(models)) {
    const color = colors[idx % colors.length];
    const count = await runInferenceWithModel(model, imgElement, color);
    outputText += `${name} 検出数: ${count}\n`;
    totalDetections += count;
    idx++;
  }

  if (totalDetections === 0) outputText = '検出結果なし';

  resultDiv.textContent = outputText;
}

runBtn.addEventListener('click', runInferenceAllModels);

loadModels();
