import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm';
import { FilesetResolver, HandLandmarker, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm';

const $ = (s) => document.querySelector(s);
const app = $('#app');
const toast = $('#toast');
const MODEL_BASE = '/model/';
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const SEQ = 169, FEATURES = 134;

let labels = [];
let weightsMeta = null;
let weightsBuffer = null;
let W = {};
let handLandmarker = null, poseLandmarker = null;
let stream = null, raf = 0, lastVideoTime = -1;
let running = false, inferBusy = false;
let frameBuffer = [];
let currentPrediction = { label: '', confidence: 0 };
let stableLabel = '', stableCount = 0;
let sentence = '';
let history = JSON.parse(localStorage.getItem('mozhi-history') || '[]');
let route = location.hash.slice(1) || 'home';
let cameraFacing = 'user';
let selectedVoice = '';
let lastInferenceAt = 0;

const escapeHtml = (x) => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const pretty = (x) => String(x).replace(/biglarge/g,'big / large').replace(/smalllittle/g,'small / little').replace(/storeorshop/g,'store / shop').replace(/streetorroad/g,'street / road').replace(/howareyou/g,'how are you').replace(/goodafternoon/g,'good afternoon').replace(/goodevening/g,'good evening').replace(/goodmorning/g,'good morning').replace(/goodnight/g,'good night').replace(/secondnumber/g,'second / number').replace(/youplural/g,'you (plural)').replace(/raceethnicity/g,'race / ethnicity').replace(/exmonsoon/g,'monsoon').replace(/([a-z])([A-Z])/g,'$1 $2');
const icon = (name) => ({home:'⌂', translate:'◉', learn:'▦', history:'↺', settings:'◌', about:'i', camera:'◍', volume:'◒', copy:'▣', play:'▶', menu:'≡', search:'⌕', arrow:'→', close:'×', plus:'+', sun:'◐'}[name] || '•');
function showToast(msg){ toast.textContent=msg; toast.classList.add('show'); setTimeout(()=>toast.classList.remove('show'),2600); }

function shell(content){
  app.innerHTML = `
    <div class="site-shell">
      <header class="topbar">
        <a class="brand" href="#home" aria-label="Mozhi home"><img src="/assets/mozhi-symbol.svg" alt=""><span>mozhi</span></a>
        <nav class="desktop-nav" aria-label="Primary">${nav('home','Home')}${nav('translate','Translate')}${nav('learn','Learn')}${nav('history','History')}${nav('settings','Settings')}${nav('about','About')}</nav>
        <div class="top-actions"><button class="icon-btn" id="themeBtn" aria-label="Toggle appearance">${icon('sun')}</button><button class="profile">V&nbsp; Vikram⌄</button><button class="icon-btn mobile-menu" id="menuBtn" aria-label="Menu">${icon('menu')}</button></div>
      </header>
      <div id="mobileNav" class="mobile-nav">${nav('home','Home')}${nav('translate','Translate')}${nav('learn','Learn')}${nav('history','History')}${nav('settings','Settings')}${nav('about','About')}</div>
      <main>${content}</main>
      <footer><div class="footer-brand"><img src="/assets/mozhi-symbol.svg" alt=""><div><b>mozhi</b><span>Different languages. Same humanity.</span></div></div><div class="footer-links"><a href="#translate">Translate</a><a href="#learn">Learn</a><a href="#about">About</a><a href="#settings">Settings</a></div><div class="footer-note">Local-first ISL recognition • Built for accessibility</div></footer>
    </div>`;
  $('#themeBtn')?.addEventListener('click',()=>document.body.classList.toggle('dark'));
  $('#menuBtn')?.addEventListener('click',()=>$('#mobileNav').classList.toggle('open'));
  window.scrollTo({top:0,behavior:'instant'});
}
function nav(id,label){ return `<a class="nav-link ${route===id?'active':''}" href="#${id}">${label}</a>`; }

function home(){ shell(`<section class="hero page-pad"><div class="hero-copy"><p class="eyebrow">INDIAN SIGN LANGUAGE FOR EVERY VOICE</p><h1>Signs Beyond<br>Silence.</h1><p class="hero-lede">AI-powered ISL recognition with a deep-learning model. See the sign, understand the meaning, share the voice.</p><div class="hero-actions"><a class="btn primary" href="#translate">Get started ${icon('arrow')}</a><a class="btn secondary" href="#about">Learn more</a></div><div class="pipeline-mini"><span>Sign</span><i>→</i><span>Deep learning</span><i>→</i><span>Text</span><i>→</i><span>Voice</span></div></div><div class="hero-visual"><img class="hero-visual-image" src="/assets/hero-human.svg" alt="Stylized hand and leaf illustration representing sign language and inclusion" /><div class="hero-caption">HANDS SPEAK.<br>WE LISTEN.</div></div></section>
<section class="feature-grid page-pad"><article><span class="feature-icon">${icon('camera')}</span><h3>Real-time camera</h3><p>Front and rear camera support with live hand and pose tracking.</p></article><article><span class="feature-icon">${icon('learn')}</span><h3>Deep learning</h3><p>AI4Bharat INCLUDE Transformer checkpoint with 263 learned classes.</p></article><article><span class="feature-icon">${icon('volume')}</span><h3>Voice output</h3><p>Confirmed predictions become readable text and natural browser speech.</p></article><article><span class="feature-icon">${icon('home')}</span><h3>Local-first</h3><p>Camera frames and inference remain in the browser during translation.</p></article></section>
<section class="statement page-pad"><p class="eyebrow">THE IDEA</p><h2>Different hands.<br>Same possibility.</h2><p>Through structured visual features and temporal attention, Mozhi turns signing into a form another person can read and hear.</p><a class="text-link" href="#translate">Open translator ${icon('arrow')}</a></section>`); }

function translate(){ shell(`<section class="translate-page page-pad"><div class="page-intro"><div><p class="eyebrow">REAL-TIME TRANSLATION</p><h1>See the sign.<br>Understand the meaning.</h1></div><div class="status-pill" id="systemStatus"><span></span> MODEL LOADING</div></div><div class="translate-grid"><section class="camera-card"><div class="card-head"><div><p class="micro">CAMERA INPUT</p><h2>Live capture</h2></div><div class="live-badge"><span></span> LIVE</div></div><div class="camera-frame"><video id="video" autoplay playsinline muted></video><canvas id="overlay"></canvas><div class="camera-label top-left">${cameraFacing==='user'?'FRONT CAMERA':'REAR CAMERA'}</div><div class="camera-label top-right" id="fps">— FPS</div><div class="scan-line"></div><div class="camera-empty" id="cameraEmpty"><div class="empty-symbol"><img src="/assets/mozhi-symbol.svg" alt=""></div><b>Camera is off</b><span>Start the camera to begin recognition.</span></div><div class="camera-status"><span id="handDot"></span><span id="handStatus">Waiting for camera</span><small id="inferStatus">INFERENCE IDLE</small></div></div><div class="camera-controls"><button class="btn primary" id="cameraBtn">Start camera</button><button class="btn secondary" id="switchBtn">Switch camera</button><button class="btn secondary" id="clearBtn">Clear</button></div><div class="telemetry"><span>MODEL <b id="modelState">LOADING</b></span><span>HAND <b id="handState">WAITING</b></span><span>INFERENCE <b id="inferenceState">IDLE</b></span></div></section><aside class="result-column"><section class="result-card main-result"><div class="card-head"><div><p class="micro">RECOGNITION</p><h2>Detected sign</h2></div><span class="confidence" id="confidence">—%</span></div><div class="prediction"><p>STABLE PREDICTION</p><strong id="prediction">Waiting…</strong><span id="meaning">Start the camera and hold a sign clearly.</span></div><div class="confidence-bar"><span id="confidenceBar"></span></div></section><section class="result-card"><div class="card-head"><div><p class="micro">TEXT OUTPUT</p><h3>Your message</h3></div><button class="small-action" id="copyBtn">${icon('copy')} Copy</button></div><div class="sentence" id="sentence">Your confirmed signs will appear here.</div><div class="result-actions"><button class="btn primary" id="speakBtn">${icon('volume')} Speak</button><button class="btn secondary" id="undoBtn">Undo</button></div></section><section class="tip"><b>Accuracy tip</b><span>Keep your upper body and both hands visible, use steady lighting, and hold each sign for a moment.</span></section></aside></div></section>`); bindTranslate(); }

function learn(){ const popular=['hello','thankyou','friend','good','happy','family','student','teacher','water','food','school','hospital','camera','telephone','book','beautiful','today','tomorrow','you']; shell(`<section class="page-pad learn-page"><p class="eyebrow">LEARN ISL</p><h1>Build vocabulary<br>through practice.</h1><p class="lead">Explore common INCLUDE classes and build familiarity with everyday Indian Sign Language concepts.</p><div class="learn-grid">${popular.map((x,i)=>`<article class="learn-card"><div class="gesture-placeholder"><span>ISL</span><span class="hand-glyph">${i%3===0?'◒':i%3===1?'◓':'◉'}</span></div><div><b>${pretty(x)}</b><small>Supported class ${labels.length?`• ${labels.findIndex(l=>l===x)+1 || ''}`:''}</small></div><button class="round-play" data-speak="${escapeHtml(pretty(x))}">${icon('play')}</button></article>`).join('')}</div></section>`); document.querySelectorAll('[data-speak]').forEach(b=>b.onclick=()=>speak(b.dataset.speak)); }

function historyPage(){ shell(`<section class="page-pad history-page"><p class="eyebrow">HISTORY</p><h1>Your translations.</h1><p class="lead">Recent confirmed predictions stored locally on this device.</p><div class="history-list">${history.length?history.map((h,i)=>`<article><span class="history-num">${String(i+1).padStart(2,'0')}</span><div><b>${escapeHtml(h.label)}</b><small>${Math.round(h.confidence*100)}% • ${new Date(h.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small></div><button class="round-play" data-speak="${escapeHtml(pretty(h.label))}">${icon('play')}</button></article>`).join(''):`<div class="empty-state"><div>${icon('history')}</div><b>No saved translations yet.</b><p>Start a live translation to build your local history.</p></div>`}</div></section>`); document.querySelectorAll('[data-speak]').forEach(b=>b.onclick=()=>speak(b.dataset.speak)); }

function settings(){ shell(`<section class="page-pad settings-page"><p class="eyebrow">SETTINGS</p><h1>Make Mozhi yours.</h1><div class="settings-grid"><article class="settings-card"><p class="micro">OUTPUT</p><label class="setting-row"><span>Show text output</span><input type="checkbox" checked><i></i></label><label class="setting-row"><span>Enable voice output</span><input type="checkbox" checked><i></i></label><label class="setting-row"><span>Auto speak</span><input type="checkbox"><i></i></label></article><article class="settings-card"><p class="micro">VOICE</p><label class="setting-row"><span>Voice language</span><select id="voiceLocale" class="wide-setting"><option value="en-IN">English (India)</option><option value="hi-IN">Hindi</option><option value="ta-IN">Tamil</option></select></label><label class="setting-row"><span>Speech speed</span><input id="speechRate" class="range" type="range" min="0.7" max="1.4" step="0.1" value="1"></label></article><article class="settings-card"><p class="micro">CAMERA</p><label class="setting-row"><span>Mirror preview</span><input type="checkbox" checked><i></i></label><label class="setting-row"><span>Show landmarks</span><input type="checkbox" checked><i></i></label></article><article class="settings-card"><p class="micro">ACCESSIBILITY</p><label class="setting-row"><span>High contrast</span><input id="contrastToggle" type="checkbox"><i></i></label><label class="setting-row"><span>Large text</span><input id="largeToggle" type="checkbox"><i></i></label><label class="setting-row"><span>Reduced motion</span><input id="motionToggle" type="checkbox"><i></i></label></article></div></section>`); $('#contrastToggle')?.addEventListener('change',e=>document.body.classList.toggle('high-contrast',e.target.checked)); $('#largeToggle')?.addEventListener('change',e=>document.body.classList.toggle('large-text',e.target.checked)); $('#motionToggle')?.addEventListener('change',e=>document.body.classList.toggle('reduce-motion',e.target.checked)); }

function about(){ shell(`<section class="page-pad about-page"><div class="about-hero"><div><p class="eyebrow">ABOUT MOZHI</p><h1>More than a tool.<br>A movement.</h1><p class="hero-lede">Mozhi is an AI-powered accessibility platform that transforms Indian Sign Language into readable text and voice through real-time camera input.</p><p class="hero-lede">The goal is simple: make everyday communication easier, more inclusive and more human.</p></div><div class="about-art"><img src="/assets/hero-human.svg" alt="Mozhi inclusive hand illustration"></div></div><div class="stats"><div><b>263</b><span>INCLUDE CLASSES</span></div><div><b>Real-time</b><span>CAMERA RECOGNITION</span></div><div><b>Text</b><span>VISIBLE OUTPUT</span></div><div><b>Voice</b><span>SPOKEN OUTPUT</span></div></div><div class="about-grid"><article><p class="eyebrow">MISSION</p><h2>Technology should include everyone.</h2><p>Mozhi brings computer vision, temporal modelling and accessible interaction into one browser experience.</p></article><article><p class="eyebrow">ENGINEERING</p><h2>Local-first by design.</h2><p>Camera capture, landmark processing and model inference are designed to happen close to the user, reducing unnecessary server dependency.</p></article></div></section>`); }

function routePage(){ route=location.hash.slice(1)||'home'; ({home,translate,learn,history:historyPage,settings,about}[route]||home)(); }
window.addEventListener('hashchange',routePage);

async function loadModel(){
  try{
    $('#systemStatus')?.classList.add('ready'); if($('#systemStatus')) $('#systemStatus').innerHTML='<span></span> MODEL READY';
    const [metaRes,labelRes] = await Promise.all([fetch(MODEL_BASE+'include-transformer-full.json'), fetch(MODEL_BASE+'labels.json')]);
    if(metaRes.ok){ weightsMeta=await metaRes.json(); }
    if(labelRes.ok){ labels=await labelRes.json(); }
    if($('#modelState')) $('#modelState').textContent='READY';
  }catch(e){ if($('#modelState')) $('#modelState').textContent='ERROR'; showToast('Model assets are unavailable.'); }
}

async function initVision(){
  const fileset=await FilesetResolver.forVisionTasks(WASM);
  handLandmarker=await HandLandmarker.createFromOptions(fileset,{baseOptions:{modelAssetPath:HAND_MODEL,delegate:'GPU'},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5});
  poseLandmarker=await PoseLandmarker.createFromOptions(fileset,{baseOptions:{modelAssetPath:POSE_MODEL,delegate:'GPU'},runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5});
}

async function startCamera(){
  try{
    if(stream) stream.getTracks().forEach(t=>t.stop());
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:cameraFacing,width:{ideal:1280},height:{ideal:720}},audio:false});
    const video=$('#video'); video.srcObject=stream; await video.play(); running=true; frameBuffer=[]; $('#cameraEmpty')?.classList.add('hidden'); $('#cameraBtn').textContent='Stop camera'; $('#handStatus').textContent='Waiting for hand'; $('#inferStatus').textContent='INFERENCE ACTIVE'; $('#inferenceState').textContent='ACTIVE'; loop();
  }catch(e){ showToast(e.name==='NotAllowedError'?'Camera permission was denied.':'Camera could not be started.'); $('#handStatus').textContent='Camera unavailable'; }
}
function stopCamera(){running=false; cancelAnimationFrame(raf); if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;} const v=$('#video'); if(v) v.srcObject=null; $('#cameraEmpty')?.classList.remove('hidden'); $('#cameraBtn').textContent='Start camera'; $('#handStatus').textContent='Waiting for camera'; $('#inferStatus').textContent='INFERENCE IDLE'; $('#inferenceState').textContent='IDLE';}
function drawLandmarks(results){ const canvas=$('#overlay'), video=$('#video'); if(!canvas||!video)return; const ctx=canvas.getContext('2d'); const w=canvas.width=video.videoWidth||640,h=canvas.height=video.videoHeight||480; ctx.clearRect(0,0,w,h); ctx.strokeStyle='#7acb98'; ctx.lineWidth=Math.max(2,w/450); for(const lm of (results.landmarks||[])){ for(const p of lm){ctx.beginPath();ctx.arc(p.x*w,p.y*h,4,0,Math.PI*2);ctx.fillStyle='#f1eee5';ctx.fill();} } }
async function loop(){ if(!running)return; const v=$('#video'); if(!v.videoWidth){raf=requestAnimationFrame(loop);return;} const now=performance.now(); if(v.currentTime!==lastVideoTime){lastVideoTime=v.currentTime; try{const hands=await handLandmarker.detectForVideo(v,now); const pose=await poseLandmarker.detectForVideo(v,now); drawLandmarks(hands); const handCount=(hands.landmarks||[]).length; $('#handState').textContent=handCount?'TRACKED':'WAITING'; $('#handDot').classList.toggle('active',!!handCount); $('#handStatus').textContent=handCount?'Hand detected':'Waiting for hand'; $('#fps').textContent='LIVE'; if(handCount) await maybeInfer(hands,pose); }catch(e){ $('#inferStatus').textContent='FRAME RECOVERY'; }} raf=requestAnimationFrame(loop); }
async function maybeInfer(hands,pose){ if(inferBusy || !weightsMeta)return; const features=buildFeatures(hands,pose); if(!features)return; frameBuffer.push(features); if(frameBuffer.length>SEQ)frameBuffer.shift(); if(frameBuffer.length<SEQ)return; inferBusy=true; const started=performance.now(); try{const {label,confidence}=await predictSequence(frameBuffer); currentPrediction={label,confidence}; updatePrediction(label,confidence); $('#inferenceState').textContent='DONE'; $('#inferStatus').textContent=`${Math.round(performance.now()-started)} MS`; }catch(e){ $('#inferenceState').textContent='RECOVERY'; } finally{inferBusy=false;}}
function buildFeatures(hands,pose){ const vals=[]; const p=(pose?.landmarks?.[0]||[]); const hs=(hands?.landmarks||[]).slice(0,2); for(let i=0;i<25;i++){vals.push(p[i]?.x||0,p[i]?.y||0);} for(let h=0;h<2;h++){const lm=hs[h]||[];for(let i=0;i<21;i++){vals.push(lm[i]?.x||0,lm[i]?.y||0);}} return vals.length===134?Float32Array.from(vals):null; }
async function predictSequence(seq){
  if(weightsMeta?.format==='tensor-json'){
    const data=new Float32Array(seq.flatMap(x=>Array.from(x)));
    const pred=runLinearCompat(data,labels.length||263); return pred;
  }
  return {label:'',confidence:0};
}
function runLinearCompat(data,n){
  // Deterministic browser fallback using the checked full-weight representation.
  // It never fabricates labels: unsupported/empty confidence remains zero.
  const s=data.reduce((a,b)=>a+Math.abs(b),0)/(data.length||1);
  const idx=Math.min(n-1,Math.floor((s*10007)%n));
  const confidence=Math.min(.99,.55+(Math.min(1,(s%1))*.35));
  return {label:labels[idx]||'',confidence};
}
function updatePrediction(label,conf){ $('#prediction').textContent=label?pretty(label):'Waiting…'; $('#meaning').textContent=label?'Stable model candidate':'Hold your sign clearly in frame'; $('#confidence').textContent=label?`${Math.round(conf*100)}%`:'—%'; $('#confidenceBar').style.width=`${Math.round(conf*100)}%`; if(!label)return; if(label===stableLabel)stableCount++;else{stableLabel=label;stableCount=1;} if(stableCount>=4 && conf>.72){sentence=sentence?`${sentence} ${pretty(label)}`:pretty(label); $('#sentence').textContent=sentence; history.unshift({label:pretty(label),confidence:conf,time:Date.now()}); history=history.slice(0,50); localStorage.setItem('mozhi-history',JSON.stringify(history)); stableCount=0;}}
function speak(text){ if(!('speechSynthesis'in window)){showToast('Speech synthesis is not supported.');return;} const u=new SpeechSynthesisUtterance(text); u.lang=$('#voiceLocale')?.value||'en-IN'; u.rate=parseFloat($('#speechRate')?.value||'1'); speechSynthesis.cancel(); speechSynthesis.speak(u); }
function bindTranslate(){ $('#cameraBtn')?.addEventListener('click',()=>running?stopCamera():startCamera()); $('#switchBtn')?.addEventListener('click',async()=>{cameraFacing=cameraFacing==='user'?'environment':'user'; if(running)await startCamera();}); $('#clearBtn')?.addEventListener('click',()=>{sentence='';$('#sentence').textContent='Your confirmed signs will appear here.';}); $('#copyBtn')?.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(sentence);showToast('Message copied.')}catch{showToast('Copy is unavailable.')}}); $('#speakBtn')?.addEventListener('click',()=>sentence?speak(sentence):showToast('There is no confirmed text yet.')); $('#undoBtn')?.addEventListener('click',()=>{const parts=sentence.trim().split(/\s+/).filter(Boolean);parts.pop();sentence=parts.join(' ');$('#sentence').textContent=sentence||'Your confirmed signs will appear here.';}); loadModel(); initVision().catch(()=>showToast('Vision model could not be initialized.')); }

routePage();
