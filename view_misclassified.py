"""
Generate HTML gallery for misclassified images from test_results_t4.csv
Opens in default browser.
"""

import csv
import webbrowser
from pathlib import Path
import base64

def image_to_base64(path):
    """Convert image to base64 for embedding in HTML."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def get_image_ext(path):
    ext = Path(path).suffix.lower()
    if ext == '.jpg':
        return 'jpeg'
    return ext[1:]

def generate_gallery(csv_path='test_results_mps.csv', output='misclassified_gallery.html'):
    # Load misclassified images
    images = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['correct'] == 'False':
                images.append(row)
    
    if not images:
        print("No misclassified images found!")
        return
    
    print(f"Found {len(images)} misclassified images")
    
    # Generate HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Misclassified Images ({len(images)})</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #1a1a1a; 
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
        }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 20px;
            color: #ff6b6b;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 30px;
            color: #888;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: #2a2a2a;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: scale(1.02);
        }}
        .card img {{
            width: 100%;
            height: 250px;
            object-fit: contain;
            background: #333;
            cursor: pointer;
        }}
        .card-info {{
            padding: 15px;
        }}
        .filename {{
            font-size: 12px;
            color: #888;
            word-break: break-all;
            margin-bottom: 8px;
        }}
        .labels {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .label {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .true-label {{
            background: #2d5a2d;
            color: #7fff7f;
        }}
        .pred-label {{
            background: #5a2d2d;
            color: #ff7f7f;
        }}
        .arrow {{
            color: #666;
            font-size: 20px;
        }}
        .confidence {{
            text-align: center;
            margin-top: 8px;
            color: #888;
            font-size: 12px;
        }}
        .split {{
            display: inline-block;
            background: #3a3a3a;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            color: #aaa;
            margin-left: 5px;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            cursor: pointer;
        }}
        .modal img {{
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
        }}
        .modal-info {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            background: rgba(0,0,0,0.8);
            padding: 15px 30px;
            border-radius: 10px;
        }}
    </style>
</head>
<body>
    <h1>🔍 Misclassified Images</h1>
    <div class="stats">{len(images)} images incorrectly classified</div>
    <div class="gallery">
'''
    
    for i, img in enumerate(images):
        try:
            b64 = image_to_base64(img['path'])
            ext = get_image_ext(img['path'])
            filename = Path(img['path']).name
            
            html += f'''
        <div class="card" onclick="openModal({i})">
            <img src="data:image/{ext};base64,{b64}" alt="{filename}">
            <div class="card-info">
                <div class="filename">{filename}<span class="split">{img['split']}</span></div>
                <div class="labels">
                    <span class="label true-label">TRUE: {img['true_label'].upper()}</span>
                    <span class="arrow">→</span>
                    <span class="label pred-label">PRED: {img['predicted'].upper()}</span>
                </div>
                <div class="confidence">Confidence: {img['confidence']}</div>
            </div>
        </div>
'''
        except Exception as e:
            print(f"Error processing {img['path']}: {e}")
    
    html += '''
    </div>
    <div class="modal" id="modal" onclick="closeModal()">
        <img id="modal-img" src="">
        <div class="modal-info" id="modal-info"></div>
    </div>
    <script>
        const images = ''' + str([{
            'src': f"data:image/{get_image_ext(img['path'])};base64,{image_to_base64(img['path'])}",
            'name': Path(img['path']).name,
            'true': img['true_label'],
            'pred': img['predicted'],
            'conf': img['confidence']
        } for img in images]) + ''';
        
        let currentIdx = 0;
        
        function openModal(idx) {
            currentIdx = idx;
            showModalImage();
            document.getElementById('modal').style.display = 'block';
        }
        
        function showModalImage() {
            const img = images[currentIdx];
            document.getElementById('modal-img').src = img.src;
            document.getElementById('modal-info').innerHTML = 
                `<strong>${img.name}</strong><br>` +
                `True: <span style="color:#7fff7f">${img.true.toUpperCase()}</span> → ` +
                `Predicted: <span style="color:#ff7f7f">${img.pred.toUpperCase()}</span> (${img.conf})<br>` +
                `<small style="color:#888">${currentIdx + 1} / ${images.length} — Use ← → keys to navigate</small>`;
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        
        document.addEventListener('keydown', (e) => {
            if (document.getElementById('modal').style.display === 'block') {
                if (e.key === 'ArrowRight' && currentIdx < images.length - 1) {
                    currentIdx++;
                    showModalImage();
                    e.preventDefault();
                } else if (e.key === 'ArrowLeft' && currentIdx > 0) {
                    currentIdx--;
                    showModalImage();
                    e.preventDefault();
                } else if (e.key === 'Escape') {
                    closeModal();
                }
            }
        });
    </script>
</body>
</html>
'''
    
    with open(output, 'w') as f:
        f.write(html)
    
    print(f"Gallery saved to {output}")
    webbrowser.open(f'file://{Path(output).absolute()}')

if __name__ == '__main__':
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_results_mps.csv'
    generate_gallery(csv_file)
