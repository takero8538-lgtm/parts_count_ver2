if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/parts_count_ver2/service-worker.js')
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

let models = {};       // { モデル名: tf.GraphModel }
let imgElement = null;
let stream = null;     // カメラ映像ストリーム保持用

// 共通：画像設定後のCanvas初期化＆runBtn有効化更新
function setImageElement(img) {
  imgElement = img;
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  runBtn.disabled = !(Object.keys(models).length > 0 && imgElement != null);
  resultDiv.textContent = '';
}

// models_list.json からモデルリストを取得
async function loadModelList() {
  try {
    const res = await fetch('models_list.json');
    const modelList = await res.json();

    if (!Array.isArray(modelList)) {
      throw new Error("models_list.json の内容が配列ではありません");
    }
    return modelList;
  } catch (error) {
    console.error("モデルリストの読み込みに失敗:", error);
    alert("モデルリストの読み込みに失敗しました");
    return [];
  }
}

// 複数モデルを一括読み込み
async function loadAllModels() {
  const modelList = await loadModelList();

  resultDiv.textContent = 'モデルを読み込み中...';
  runBtn.disabled = true;
  models = {};

  for (const m of modelList) {
    try {
      const model = await tf.loadGraphModel(m.path + 'model.json');
      models[m.name] = model;
      console.log(`${m.name} 読み込み成功`);
    } catch (e) {
      console.warn(`${m.name} の読み込みをスキップ（${e.message}）`);
    }
  }

  const loadedCount = Object.keys(models).length;
  if (loadedCount === 0) {
    resultDiv.textContent = '【エラー】すべてのモデルの読み込みに失敗しました。';
    runBtn.disabled = true;
    alert('モデルが１つも読み込めませんでした。環境を確認してください。');
  } else {
    resultDiv.textContent = `モデル${loadedCount}件が読み込み完了しました。画像を選択または撮影してください。`;
    runBtn.disabled = !(imgElement != null);
  }
}

// 単一モデル推論＆NMS処理後の描画
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

    const confThreshold = 0.2;

    // 推論結果からボックスとスコアを抽出
    for (let i = 0; i < numBoxes; i++) {
      const offset = i * numAttributes;

      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      let maxScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = data[offset + 4 + c];
        if (score > maxScore) {
          maxScore = score;
        }
      }

      if (maxScore >= confThreshold) {
        const ymin = cy - h / 2;
        const xmin = cx - w / 2;
        const ymax = cy + h / 2;
        const xmax = cx + w / 2;
        boxes.push([ymin, xmin, ymax, xmax]);
        scores.push(maxScore);
      }
    }

    let indices = [];

    if (boxes.length > 0) {
      const boxesTensor = tf.tensor2d(boxes);
      const scoresTensor = tf.tensor1d(scores);

      const maxOutputSize = 300;
      const iouThreshold = 0.45;

      const selectedIndices = await tf.image.nonMaxSuppressionAsync(
        boxesTensor,
        scoresTensor,
        maxOutputSize,
        iouThreshold,
        confThreshold
      );

      indices = await selectedIndices.data();

      boxesTensor.dispose();
      scoresTensor.dispose();
      selectedIndices.dispose();
    }

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;

    // NMSで選ばれたボックスのみ描画
    for (const i of indices) {
      const [ymin, xmin, ymax, xmax] = boxes[i];

      const realXmin = (xmin - padLeft) / scale;
      const realYmin = (ymin - padTop) / scale;
      const realXmax = (xmax - padLeft) / scale;
      const realYmax = (ymax - padTop) / scale;

      const boxWidth = realXmax - realXmin;
      const boxHeight = realYmax - realYmin;

      if (boxWidth > 0 && boxHeight > 0) {
        ctx.strokeRect(realXmin, realYmin, boxWidth, boxHeight);
      }
    }

    squeezed.dispose();
    transposed.dispose();
    rawOutput.dispose();
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);

    return { count: indices.length };
  } catch (error) {
    console.error(error);
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);
    return { count: 0, error: error.message };
  }
}

// 複数モデルを直列で推論し、検出数が1件以上のモデルのみ表示
async function runInferenceAllModels() {
  if (!imgElement || Object.keys(models).length === 0) {
    alert('画像またはモデルがありません。');
    return;
  }

  canvas.width = imgElement.width;
  canvas.height = imgElement.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgElement, 0, 0);

  let resultText = '';
  let idx = 0;
  for (const [name, model] of Object.entries(models)) {
    const color = colors[idx % colors.length];
    const res = await runInferenceWithModel(model, imgElement, color);

    if (res.error) {
      resultText += `${name}: エラー (${res.error})\n`;
    } else if (res.count > 0) {
      resultText += `${name}: 検出数 ${res.count}\n`;
    }
    idx++;
  }

  if (resultText === '') {
    resultText = '検出結果なし';
  }

  resultDiv.textContent = resultText;
}

// カメラ停止処理
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

// 画像ファイル選択時処理
imageInput.addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (!file) return;

  stopCamera();

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => setImageElement(img);
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

// カメラ起動／停止ボタン
// カメラ起動／停止ボタン
startCameraBtn.addEventListener('click', async () => {
  if (stream) {
    stopCamera();
    return;
  }

  // 【修正】以前の選択画像・ファイル選択状態をリセット
  imgElement = null;
  runBtn.disabled = true;
  imageInput.value = ''; 

  // 【修正】Canvasをクリアした上で、サイズを0にして完全に消す
  canvas.width = 0;
  canvas.height = 0;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
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
// 写真撮影ボタン
captureBtn.addEventListener('click', () => {
  if (!stream) return;

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

  const img = new Image();
  img.src = canvas.toDataURL('image/jpeg');
  img.onload = () => {
    setImageElement(img);
    resultDiv.textContent = '写真を撮影しました。「カウント開始」を押してください。';
    stopCamera();
  };
});

// 推論実行ボタン
runBtn.addEventListener('click', runInferenceAllModels);

// ページ読み込み時にモデルまとめて読み込み開始
loadAllModels();
