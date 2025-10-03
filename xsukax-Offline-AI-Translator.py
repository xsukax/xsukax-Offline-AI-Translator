"""
xsukax Offline AI Translator
Facebook NLLB-200 multilingual translator supporting 50 languages
"""

import os
import sys
import json
import threading
import time
import re
from flask import Flask, render_template_string, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Disable Windows symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(APP_DIR, 'models')
SETTINGS_FILE = os.path.join(APP_DIR, 'settings.json')

os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

app = Flask(__name__)

model = None
tokenizer = None
selected_model_name = None
loading_status = {"loading": False, "progress": 0, "message": "", "complete": False}
lang_token_map = {}

AVAILABLE_MODELS = {
    "1": {"name": "facebook/nllb-200-distilled-600M", "display": "NLLB-200-600M (Fast)", "desc": "Smallest, fastest", "size": 600},
    "2": {"name": "facebook/nllb-200-1.3B", "display": "NLLB-200-1.3B (Recommended)", "desc": "Best balance", "size": 1300},
    "3": {"name": "facebook/nllb-200-distilled-1.3B", "display": "NLLB-200-1.3B Distilled", "desc": "Fast + quality", "size": 1300},
    "4": {"name": "facebook/nllb-200-3.3B", "display": "NLLB-200-3.3B (Best)", "desc": "Highest quality", "size": 3300}
}

LANGUAGES = {
    "eng_Latn": "English",
    "arb_Arab": "Arabic",
    "amh_Ethi": "Amharic",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "ita_Latn": "Italian",
    "tur_Latn": "Turkish",
    "pol_Latn": "Polish",
    "ukr_Cyrl": "Ukrainian",
    "nld_Latn": "Dutch",
    "ell_Grek": "Greek",
    "swe_Latn": "Swedish",
    "ces_Latn": "Czech",
    "ron_Latn": "Romanian",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "urd_Arab": "Urdu",
    "pes_Arab": "Persian",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "tgl_Latn": "Filipino",
    "swh_Latn": "Swahili",
    "heb_Hebr": "Hebrew",
    "dan_Latn": "Danish",
    "fin_Latn": "Finnish",
    "nob_Latn": "Norwegian",
    "hun_Latn": "Hungarian",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "som_Latn": "Somali",
    "hau_Latn": "Hausa",
    "yor_Latn": "Yoruba",
    "zul_Latn": "Zulu",
    "afr_Latn": "Afrikaans",
    "bul_Cyrl": "Bulgarian",
    "hrv_Latn": "Croatian",
    "slk_Latn": "Slovak"
}

def save_settings(model_choice):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({'model_choice': model_choice}, f)
        return True
    except:
        return False

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f).get('model_choice')
    except:
        pass
    return None

def get_folder_size(folder_path):
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total

def display_model_menu():
    print("\n" + "="*60)
    print("SELECT MODEL")
    print("="*60)
    
    for key, info in AVAILABLE_MODELS.items():
        print(f"[{key}] {info['display']}")
        print(f"    {info['desc']}\n")
    
    saved = load_settings()
    if saved and saved in AVAILABLE_MODELS:
        print(f"Last used: [{saved}] {AVAILABLE_MODELS[saved]['display']}")
        print("Press Enter to reuse, or type number to change")
    
    while True:
        try:
            choice = input(f"\nChoice [1-4]{' or Enter' if saved else ''}: ").strip()
            choice = saved if choice == "" and saved else choice
            
            if choice in AVAILABLE_MODELS:
                selected = AVAILABLE_MODELS[choice]
                print(f"\nSelected: {selected['display']}")
                save_settings(choice)
                return selected['name'], selected['display'], selected['size']
            else:
                print("Invalid. Enter 1-4.")
        except KeyboardInterrupt:
            print("\n\nCancelled")
            sys.exit(0)

def load_model_with_progress(model_name, display_name, expected_size_mb):
    global model, tokenizer, selected_model_name, loading_status, lang_token_map
    
    if model is not None:
        loading_status["complete"] = True
        loading_status["message"] = "Model already loaded"
        return
    
    selected_model_name = display_name
    loading_status["loading"] = True
    loading_status["progress"] = 0
    loading_status["message"] = "Initializing..."
    
    try:
        loading_status["message"] = "Loading tokenizer..."
        loading_status["progress"] = 10
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        print("\n✓ Tokenizer loaded")
        print(f"  Type: {type(tokenizer).__name__}")
        
        print("\n  Building language token map...")
        vocab = tokenizer.get_vocab()
        
        for token, token_id in vocab.items():
            if '_' in token and len(token.split('_')) == 2:
                parts = token.split('_')
                if len(parts[0]) == 3 and len(parts[1]) == 4:
                    lang_token_map[token] = token_id
        
        print(f"  ✓ Found {len(lang_token_map)} language tokens")
        
        missing = sum(1 for lang in LANGUAGES.keys() if lang not in lang_token_map)
        if missing == 0:
            print(f"  ✓ All {len(LANGUAGES)} languages validated")
        
        loading_status["message"] = "Loading model..."
        loading_status["progress"] = 30
        
        initial_size = get_folder_size(MODEL_CACHE_DIR)
        expected_bytes = expected_size_mb * 1024 * 1024
        
        def monitor_progress():
            while loading_status["loading"] and loading_status["progress"] < 90:
                time.sleep(1)
                current_size = get_folder_size(MODEL_CACHE_DIR)
                downloaded = current_size - initial_size
                if expected_bytes > 0:
                    progress = 30 + int((downloaded / expected_bytes) * 60)
                    loading_status["progress"] = min(progress, 90)
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        print("✓ Model loaded")
        
        loading_status["progress"] = 100
        loading_status["message"] = "Ready!"
        loading_status["loading"] = False
        loading_status["complete"] = True
        
    except Exception as e:
        loading_status["loading"] = False
        loading_status["message"] = f"Error: {str(e)}"
        loading_status["progress"] = 0
        print(f"\n✗ Error: {e}")
        raise

def load_model(model_name, display_name, expected_size_mb):
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"Model: {display_name}")
    print(f"Cache: {MODEL_CACHE_DIR}\n")
    
    thread = threading.Thread(target=load_model_with_progress, args=(model_name, display_name, expected_size_mb))
    thread.start()
    
    last_progress = -1
    while not loading_status["complete"] and not loading_status["message"].startswith("Error"):
        if loading_status["progress"] != last_progress:
            print(f"\r{loading_status['message']} [{loading_status['progress']}%]", end="", flush=True)
            last_progress = loading_status["progress"]
        time.sleep(0.3)
        if loading_status["complete"]:
            break
    
    thread.join()
    
    if loading_status["message"].startswith("Error"):
        raise Exception(loading_status["message"])
    
    print(f"\n\n{'='*60}")
    print("✓ READY TO TRANSLATE")
    print("="*60)
    print(f"Model: {display_name}")
    print(f"Languages: {len(LANGUAGES)} supported")
    print(f"Max input: 512 tokens per segment\n")

def split_by_newlines(text):
    """Split text by newlines while preserving the newline structure"""
    segments = []
    current_segment = []
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        if line.strip():
            current_segment.append(line)
        else:
            if current_segment:
                segments.append('\n'.join(current_segment))
                current_segment = []
            segments.append('')
    
    if current_segment:
        segments.append('\n'.join(current_segment))
    
    return segments

def translate_segment(segment, source_lang, target_lang):
    """Translate a single segment"""
    global model, tokenizer, lang_token_map
    
    if not segment.strip():
        return segment
    
    tokenizer.src_lang = source_lang
    inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True, max_length=512)
    target_token_id = lang_token_map[target_lang]
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

def translate_text(text, source_lang, target_lang):
    """Translate text while preserving newline structure"""
    global model, tokenizer, lang_token_map
    
    if model is None or tokenizer is None:
        raise Exception("Model not loaded")
    
    print(f"\n{'='*50}")
    print(f"TRANSLATION")
    print(f"{'='*50}")
    print(f"From: {LANGUAGES.get(source_lang, source_lang)}")
    print(f"To: {LANGUAGES.get(target_lang, target_lang)}")
    print(f"Text length: {len(text)} chars")
    
    if source_lang not in lang_token_map:
        raise Exception(f"Source language not supported: {source_lang}")
    
    if target_lang not in lang_token_map:
        raise Exception(f"Target language not supported: {target_lang}")
    
    try:
        # Split text by newlines
        segments = split_by_newlines(text)
        print(f"Segments: {len(segments)}")
        
        # Translate each segment
        translated_segments = []
        for i, segment in enumerate(segments):
            if segment.strip():
                translated = translate_segment(segment, source_lang, target_lang)
                translated_segments.append(translated)
                print(f"  Segment {i+1}/{len(segments)}: {len(segment)} -> {len(translated)} chars")
            else:
                translated_segments.append('')
        
        # Reconstruct with newlines
        result = '\n'.join(translated_segments)
        
        print(f"✓ Translation complete ({len(result)} chars)")
        print(f"{'='*50}\n")
        
        if not result.strip():
            raise Exception("Empty translation")
        
        return result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"{'='*50}\n")
        raise

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>xsukax Offline AI Translator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; line-height: 1.5; }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
        header { background: #fff; border-bottom: 1px solid #ddd; padding: 16px 0; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .header-content { text-align: center; }
        .logo { font-size: 24px; font-weight: 700; color: #2d7a3e; }
        h1 { font-size: 20px; font-weight: 600; margin-top: 4px; }
        .subtitle { color: #666; font-size: 13px; margin-top: 4px; }
        .model-info { color: #999; font-size: 11px; margin-top: 2px; font-style: italic; }
        .translator-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .language-selector { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; margin-bottom: 20px; align-items: center; }
        @media (max-width: 768px) { .language-selector { grid-template-columns: 1fr; gap: 12px; } .swap-btn-container { order: 3; } }
        .language-dropdown { position: relative; }
        .dropdown-label { font-size: 11px; font-weight: 600; color: #666; margin-bottom: 6px; display: block; text-transform: uppercase; }
        select { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; background: #fff; cursor: pointer; }
        select:focus { outline: none; border-color: #2d7a3e; }
        select:hover { border-color: #2d7a3e; }
        .swap-btn-container { display: flex; justify-content: center; }
        .swap-btn { padding: 10px; border: 1px solid #ddd; background: #fff; border-radius: 50%; cursor: pointer; font-size: 18px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .swap-btn:hover { background: #2d7a3e; color: #fff; border-color: #2d7a3e; transform: rotate(180deg); }
        .translation-area { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        @media (max-width: 768px) { .translation-area { grid-template-columns: 1fr; } }
        .text-panel { display: flex; flex-direction: column; }
        .panel-label { font-size: 13px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; }
        .char-count { font-size: 11px; color: #666; font-weight: 400; margin-left: auto; }
        textarea { width: 100%; min-height: 200px; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; resize: vertical; font-family: inherit; white-space: pre-wrap; }
        textarea:focus { outline: none; border-color: #2d7a3e; }
        textarea:disabled { background: #f9f9f9; color: #333; }
        .btn-group { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
        .btn { padding: 10px 20px; border: 1px solid #ddd; background: #f5f5f5; color: #333; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s; }
        .btn:hover:not(:disabled) { background: #fff; border-color: #2d7a3e; }
        .btn-primary { background: #2d7a3e; color: #fff; border-color: #2d7a3e; }
        .btn-primary:hover:not(:disabled) { background: #256932; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .copy-btn { padding: 6px 12px; font-size: 12px; margin-top: 8px; align-self: flex-start; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center; }
        .modal.show { display: flex; }
        .modal-content { background: #fff; border-radius: 8px; padding: 24px; max-width: 400px; width: 90%; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .modal-header { font-size: 18px; font-weight: 600; margin-bottom: 12px; }
        .modal-body { color: #666; font-size: 14px; margin-bottom: 20px; }
        .modal-footer { display: flex; justify-content: flex-end; }
        .status { margin-top: 16px; padding: 12px; border-radius: 6px; font-size: 14px; display: none; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .loading { display: none; text-align: center; margin: 16px 0; }
        .loading.show { display: block; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #ddd; border-top-color: #2d7a3e; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .progress-container { margin: 16px 0; display: none; }
        .progress-container.show { display: block; }
        .progress-bar-bg { width: 100%; height: 24px; background: #e9ecef; border-radius: 12px; overflow: hidden; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #2d7a3e, #3a9d4f); transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 12px; font-weight: 600; }
        .progress-message { text-align: center; margin-top: 8px; font-size: 13px; color: #666; }
        footer { text-align: center; color: #999; font-size: 12px; margin-top: 24px; padding-top: 16px; border-top: 1px solid #ddd; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin: 0 2px; }
        .badge-offline { background: #d1ecf1; color: #0c5460; }
        .badge-quality { background: #d4edda; color: #155724; }
        .badge-lang { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="header-content">
                <div class="logo">xsukax Offline AI Translator</div>
                <h1>NLLB-200 Neural Translation</h1>
                <div class="subtitle">
                    <span class="badge badge-offline">100% OFFLINE</span>
                    <span class="badge badge-quality">HIGH QUALITY</span>
                    <span class="badge badge-lang">50 LANGUAGES</span>
                </div>
                <div class="model-info">{{ model_name }}</div>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="translator-card">
            <div class="progress-container" id="model-progress">
                <div class="progress-bar-bg">
                    <div class="progress-bar" id="progress-bar" style="width: 0%;">0%</div>
                </div>
                <div class="progress-message" id="progress-message">Loading model...</div>
            </div>

            <div class="language-selector">
                <div class="language-dropdown">
                    <label class="dropdown-label">From</label>
                    <select id="source-lang">{{ language_options }}</select>
                </div>
                
                <div class="swap-btn-container">
                    <button class="swap-btn" onclick="swapLanguages()" title="Swap languages">⇄</button>
                </div>
                
                <div class="language-dropdown">
                    <label class="dropdown-label">To</label>
                    <select id="target-lang">{{ language_options }}</select>
                </div>
            </div>

            <div class="translation-area">
                <div class="text-panel">
                    <div class="panel-label">
                        <span>Source Text</span>
                        <span class="char-count" id="source-count">0 / 5000</span>
                    </div>
                    <textarea id="source-text" placeholder="Enter text (preserves newlines and paragraphs)..."></textarea>
                    <button class="btn copy-btn" onclick="copyText('source')">Copy</button>
                </div>

                <div class="text-panel">
                    <div class="panel-label">
                        <span>Translation</span>
                        <span class="char-count" id="target-count">0</span>
                    </div>
                    <textarea id="target-text" placeholder="Translation appears here..." disabled></textarea>
                    <button class="btn copy-btn" onclick="copyText('target')">Copy</button>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div style="margin-top: 8px; color: #666;">Translating...</div>
            </div>

            <div class="btn-group">
                <button class="btn btn-primary" onclick="performTranslation()" id="translate-btn">Translate</button>
                <button class="btn" onclick="clearAll()">Clear</button>
            </div>

            <div class="status" id="status"></div>
        </div>

        <footer>
            <p><strong>xsukax</strong> Offline AI Translator | NLLB-200</p>
            <p style="margin-top: 4px;">50 languages - Preserves formatting - All processing local</p>
        </footer>
    </div>

    <div class="modal" id="modal">
        <div class="modal-content">
            <div class="modal-header" id="modal-header">Notification</div>
            <div class="modal-body" id="modal-body">Message</div>
            <div class="modal-footer">
                <button class="btn btn-primary" onclick="closeModal()">OK</button>
            </div>
        </div>
    </div>

    <script>
        let modelReady = false;
        let progressCheckInterval = null;

        function checkModelStatus() {
            fetch('/model_status')
                .then(res => res.json())
                .then(data => {
                    if (data.loading) {
                        document.getElementById('model-progress').classList.add('show');
                        document.getElementById('progress-bar').style.width = data.progress + '%';
                        document.getElementById('progress-bar').textContent = data.progress + '%';
                        document.getElementById('progress-message').textContent = data.message;
                        document.getElementById('translate-btn').disabled = true;
                    } else if (data.complete) {
                        document.getElementById('model-progress').classList.remove('show');
                        document.getElementById('translate-btn').disabled = false;
                        modelReady = true;
                        if (progressCheckInterval) {
                            clearInterval(progressCheckInterval);
                            progressCheckInterval = null;
                        }
                    }
                })
                .catch(err => console.error('Status check failed:', err));
        }

        progressCheckInterval = setInterval(checkModelStatus, 500);
        checkModelStatus();

        function updateCharCount() {
            const text = document.getElementById('source-text').value;
            const count = document.getElementById('source-count');
            count.textContent = text.length + ' / 5000';
            count.style.color = text.length > 5000 ? '#d32f2f' : text.length > 4500 ? '#f57c00' : '#666';
        }

        document.getElementById('source-text').addEventListener('input', updateCharCount);

        function swapLanguages() {
            const source = document.getElementById('source-lang');
            const target = document.getElementById('target-lang');
            const sourceText = document.getElementById('source-text');
            const targetText = document.getElementById('target-text');
            
            const tempLang = source.value;
            source.value = target.value;
            target.value = tempLang;
            
            if (targetText.value) {
                const tempText = sourceText.value;
                sourceText.value = targetText.value;
                targetText.value = tempText;
                updateCharCount();
                document.getElementById('target-count').textContent = targetText.value.length;
            }
        }

        function performTranslation() {
            const text = document.getElementById('source-text').value;
            const sourceLang = document.getElementById('source-lang').value;
            const targetLang = document.getElementById('target-lang').value;
            
            if (!modelReady) {
                showStatus('Model is still loading, please wait...', 'info');
                return;
            }

            if (!text.trim()) {
                showStatus('Enter text to translate', 'error');
                return;
            }

            if (text.length > 5000) {
                showStatus('Text exceeds 5000 characters', 'error');
                return;
            }

            if (sourceLang === targetLang) {
                showStatus('Source and target must be different', 'error');
                return;
            }

            document.getElementById('loading').classList.add('show');
            document.getElementById('translate-btn').disabled = true;
            document.getElementById('status').style.display = 'none';

            fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: text, 
                    source_lang: sourceLang, 
                    target_lang: targetLang 
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showStatus(data.error, 'error');
                } else if (data.translation) {
                    document.getElementById('target-text').value = data.translation;
                    document.getElementById('target-count').textContent = data.translation.length;
                    showStatus('Translation complete', 'success');
                } else {
                    showStatus('No translation received', 'error');
                }
            })
            .catch(error => {
                console.error('Translation error:', error);
                showStatus('Translation failed', 'error');
            })
            .finally(() => {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('translate-btn').disabled = false;
            });
        }

        function clearAll() {
            document.getElementById('source-text').value = '';
            document.getElementById('target-text').value = '';
            document.getElementById('source-count').textContent = '0 / 5000';
            document.getElementById('source-count').style.color = '#666';
            document.getElementById('target-count').textContent = '0';
            document.getElementById('status').style.display = 'none';
        }

        function copyText(type) {
            const text = document.getElementById(type + '-text').value;
            
            if (!text) {
                showStatus('No text to copy', 'error');
                return;
            }

            navigator.clipboard.writeText(text)
                .then(() => showStatus('Copied to clipboard', 'success'))
                .catch(() => showStatus('Copy failed', 'error'));
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            setTimeout(() => status.style.display = 'none', 4000);
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('show');
        }

        document.getElementById('modal').addEventListener('click', e => {
            if (e.target.id === 'modal') closeModal();
        });

        document.getElementById('source-text').addEventListener('keydown', e => {
            if (e.ctrlKey && e.key === 'Enter') performTranslation();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    options = [f'<option value="{code}" {"selected" if code == "eng_Latn" else ""}>{name}</option>' 
               for code, name in sorted(LANGUAGES.items(), key=lambda x: x[1])]
    
    html = HTML_TEMPLATE.replace('{{ language_options }}', '\n'.join(options))
    html = html.replace('{{ model_name }}', f"Using: {selected_model_name}" if selected_model_name else "Loading...")
    html = html.replace('value="arb_Arab" >', 'value="arb_Arab" selected>', 1)
    
    return html

@app.route('/model_status', methods=['GET'])
def model_status():
    return jsonify(loading_status)

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    try:
        if model is None or tokenizer is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'eng_Latn')
        target_lang = data.get('target_lang', 'arb_Arab')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text exceeds 5000 characters'}), 400
        
        if source_lang == target_lang:
            return jsonify({'error': 'Languages must be different'}), 400
        
        if source_lang not in LANGUAGES or target_lang not in LANGUAGES:
            return jsonify({'error': 'Unsupported language'}), 400
        
        translation = translate_text(text, source_lang, target_lang)
        
        return jsonify({
            'translation': translation,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'success': True
        })
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ API Error: {error_msg}\n")
        return jsonify({'error': error_msg}), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify({'languages': LANGUAGES})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("xsukax Offline AI Translator v3.2")
    print("="*60)
    print("\nFeatures:")
    print("- 4 NLLB-200 model variants")
    print("- 50 languages supported")
    print("- Preserves newlines and paragraphs")
    print("- Max 5000 characters (auto-segmented)")
    print("- 100% offline after download\n")
    
    chosen_model, model_display, expected_size = display_model_menu()
    
    print(f"\nApp: {APP_DIR}")
    print(f"Cache: {MODEL_CACHE_DIR}")
    
    try:
        load_model(chosen_model, model_display, expected_size)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFailed to load model: {e}")
        sys.exit(1)
    
    print("="*60)
    print("Server: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        sys.exit(0)