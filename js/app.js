if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(reg => console.log('Service Worker 登録成功:', reg))
      .catch(err => console.error('Service Worker 登録失敗:', err));
  });
}

// HTML要素参照（modelSelectは廃止したので除外）
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const video = document.getElementById('video');

let models = {};  // 複数モデルここに格納
let imgElement = null;
let stream = null;

// 画像/カメラ周りは既存のまま
// --- imageInput, startCameraBtn, captureBtn, stopCamera() など ---

// 単一モデル読み込みを複数モデル読み込みに変更
async function loadAllModels() {
  const modelList = [
    { name: 'モデルA', path: 'models/modelA/' },
    { name: 'モデルB', path: 'models/modelB/' },
    { name: 'モデルC', path: 'models/modelC/' },
    // 必要に応じてモデルを増やす
  ];

  resultDiv.textContent = 'モデルを読み込み中...';
  runBtn.disabled = true;

  for (const m of modelList) {
    try {
      const model = await tf.loadGraphModel(m.path + 'model.json');
      models[m.name] = model;
      console.log(`${m.name} 読み込み成功`);
    } catch (e) {
      console.error(`${m.name} の読み込み失敗`, e);
    }
  }

  const loadedCount = Object.keys(models).length;
  if (loadedCount > 0) {
    resultDiv.textContent = `モデル${loadedCount}件が読み込み完了しました。画像を選択または撮影してください。`;
    runBtn.disabled = !(imgElement != null);
  } else {
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
  }
}

// 既存のrunInferenceをモデル指定版に変更して再利用
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

    let count = boxes.length;
    let maxConfidence = scores.length > 0 ? Math.max(...scores) : 0;

    // 描画
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.font = '16px Arial';
    ctx.fillStyle = color;

    for (let i = 0; i < count; i++) {
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
        // ラベル表示も追加可能
        // ctx.fillText(`ID:${classId} ${(score * 100).toFixed(1)}%`, realXmin + 5, realYmin + 18);
      }
    }

    squeezed.dispose();
    transposed.dispose();
    rawOutput.dispose();
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);

    return { count, maxConfidence };
  } catch (error) {
    console.error(error);
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);
    return { count: 0, maxConfidence: 0, error: error.message };
  }
}

// すべてのモデルで推論し結果をまとめて表示
async function runInferenceAllModels() {
  if (!imgElement || Object.keys(models).length === 0) {
    alert('画像またはモデルがありません。');
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgElement, 0, 0);

  const colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']; // モデルごとの色割当

  let resultText = '';

  let idx = 0;
  for (const [name, model] of Object.entries(models)) {
    const color = colors[idx % colors.length];
    const res = await runInferenceWithModel(model, imgElement, color);

    if (res.error) {
      resultText += `${name}: エラー (${res.error})\n`;
    } else {
      resultText += `${name}: 検出数 ${res.count}, 最高信頼度 ${(res.maxConfidence * 100).toFixed(1)}%\n`;
    }
    idx++;
  }

  resultDiv.textContent = resultText;
}

imageInput.addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (!file) return;

  stopCamera(); // カメラ停止

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      imgElement = img;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = !(Object.keys(models).length > 0 && imgElement != null);
      resultDiv.textContent = '';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

runBtn.addEventListener('click', runInferenceAllModels);

// 他のcamera関連イベント（startCameraBtn, captureBtn, stopCamera）は省略せずにこれまでと同様に使用

loadAllModels();
