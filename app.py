"""
Floor Plan Analysis Web Application
A user-friendly interface for analyzing floor plans
"""

import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from pathlib import Path
import tempfile
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Floor Plan Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)


class OllamaAssistant:
    """AI Assistant using Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434", model="gemma3"):
        self.base_url = base_url
        self.model = model
        self.chat_history = []
    
    def check_status(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def list_installed_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return []
            return [m.get("name", "").split(":")[0] for m in response.json().get("models", [])]
        except:
            return []
    
    def resolve_model(self):
        """Match short name (e.g. gemma3) to installed tag (e.g. gemma3:latest)."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return self.model
            for entry in response.json().get("models", []):
                name = entry.get("name", "")
                if name == self.model or name.startswith(f"{self.model}:"):
                    return name
            return self.model
        except:
            return self.model
    
    def _format_error(self, response):
        try:
            detail = response.json().get("error", "")
        except Exception:
            detail = response.text[:300] if response.text else ""
        
        if response.status_code == 404:
            return (
                f"Model `{self.model}` not found (404).\n\n"
                f"Run in terminal: `ollama pull {self.model}`"
            )
        
        gpu_crash_markers = (
            "device kernel image is invalid",
            "CUDA error",
            "0xc0000409",
            "stack-based buffer",
            "llama-server process has terminated",
        )
        if any(m.lower() in detail.lower() for m in gpu_crash_markers):
            return (
                "Ollama GPU/CUDA error — llama-server crashed (often GPU driver mismatch).\n\n"
                "**Fix (pick one):**\n"
                "1. **CPU mode (quick):** quit Ollama from the system tray, then run "
                "`start_ollama_cpu.bat` in the project folder and refresh this page.\n"
                "2. **Update GPU driver:** install the latest NVIDIA driver from nvidia.com.\n"
                "3. **Try another model:** `ollama pull llama3.2` then select it above.\n\n"
                f"Details: {detail[:200]}"
            )
        
        if detail:
            return f"Ollama error ({response.status_code}): {detail[:400]}"
        return f"Ollama error: HTTP {response.status_code}"
    
    def create_context(self, results):
        rooms = results.get('rooms', [])
        furniture = results.get('furniture', [])
        furniture_counts = Counter([f['class'] for f in furniture])
        total_area = sum(r['area_m2'] for r in rooms)
        
        context = f"""Floor Plan Analysis:
- Rooms: {len(rooms)}
- Total Area: {total_area:.2f} m²
- Furniture: {len(furniture)} items

Details:
"""
        for i, room in enumerate(rooms[:5], 1):
            context += f"Room {i}: {room['area_m2']:.2f} m²\n"
        
        for item, count in sorted(furniture_counts.items()):
            context += f"{item}: {count}\n"
        
        return context
    
    def chat(self, message, context=None):
        try:
            if context and len(self.chat_history) == 0:
                self.chat_history.append({
                    "role": "system",
                    "content": f"You are an interior design expert. Analyze this floor plan:\n{context}\n\nProvide helpful, concise advice."
                })
            
            self.chat_history.append({"role": "user", "content": message})
            
            model = self.resolve_model()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": self.chat_history, "stream": False},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['message']['content']
                self.chat_history.append({"role": "assistant", "content": answer})
                return answer
            else:
                self.chat_history.pop()
                return self._format_error(response)
        except requests.exceptions.Timeout:
            return "Timeout. Please try again."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Is it running?"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def reset(self):
        self.chat_history = []


class FloorPlanAnalyzer:
    """Complete Floor Plan Analysis System"""
    
    def __init__(self, detection_model_path, segmentation_model_path):
        """Initialize models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load detection model
        self.detector = YOLO(detection_model_path)
        
        # Load segmentation model
        num_classes = 5
        self.segmenter = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=num_classes,
            activation=None
        )
        self.segmenter.load_state_dict(
            torch.load(segmentation_model_path, map_location=self.device)
        )
        self.segmenter = self.segmenter.to(self.device)
        self.segmenter.eval()
        
        # Class names and colors
        self.seg_class_names = {
            0: 'Background',
            1: 'Wall',
            2: 'Room'
        }
        
        self.seg_colors = {
            0: [0, 0, 0],
            1: [128, 128, 128],
            2: [0, 255, 0]
        }
    
    def segment_image(self, image):
        """Perform room segmentation"""
        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        
        original_h, original_w = image.shape[:2]
        
        # Transform
        transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.segmenter(image_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def extract_rooms(self, mask):
        """Extract individual rooms"""
        room_mask = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_mask, connectivity=8
        )
        
        rooms = []
        for i in range(1, num_labels):
            area_pixels = stats[i, cv2.CC_STAT_AREA]
            if area_pixels > 100:  # Filter small noise
                rooms.append({
                    'id': i,
                    'area_pixels': int(area_pixels),
                    'centroid': (int(centroids[i][0]), int(centroids[i][1]))
                })
        
        rooms.sort(key=lambda r: r['area_pixels'], reverse=True)
        return rooms
    
    def detect_furniture(self, image_path, conf_threshold=0.25):
        """Detect furniture with bounding boxes"""
        results = self.detector.predict(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"Total detections: {len(boxes)}")  # Debug
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()  # Get bounding box coordinates
                class_name = self.detector.names[cls_id]
                
                print(f"Detected: {class_name}, Confidence: {conf:.2f}")  # Debug
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]  # [x1, y1, x2, y2]
                })
        
        return detections
    
    def create_colored_mask(self, mask):
        """Create colored visualization"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.seg_colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def calculate_areas(self, rooms, pixels_per_cm):
        """Calculate room areas"""
        for room in rooms:
            area_cm2 = room['area_pixels'] / (pixels_per_cm ** 2)
            room['area_m2'] = area_cm2 / 10000
        
        return rooms
    
    def analyze(self, image, image_path, ref_pixels, ref_length_cm):
        """Complete analysis"""
        # Calibrate
        pixels_per_cm = ref_pixels / ref_length_cm
        
        # Segment
        mask = self.segment_image(image)
        rooms = self.extract_rooms(mask)
        
        # Calculate areas
        rooms = self.calculate_areas(rooms, pixels_per_cm)
        
        # Detect furniture (with lower confidence threshold to catch more items)
        furniture = self.detect_furniture(image_path, conf_threshold=0.15)
        
        # Create visualizations
        colored_mask = self.create_colored_mask(mask)
        overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
        
        # Draw room labels
        for i, room in enumerate(rooms[:10], 1):  # Top 10 rooms
            cx, cy = room['centroid']
            cv2.circle(overlay, (cx, cy), 8, (255, 255, 0), -1)
            text = f"R{i}: {room['area_m2']:.1f}sqm"  # Use 'sqm' instead of m²
            cv2.putText(overlay, text, (cx - 50, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'rooms': rooms,
            'furniture': furniture,
            'mask': mask,
            'overlay': overlay,
            'colored_mask': colored_mask
        }


@st.cache_resource
def load_analyzer():
    """Load analyzer with caching"""
    try:
        analyzer = FloorPlanAnalyzer(
            detection_model_path='runs/detect/train_90/weights/best.pt',
            segmentation_model_path='models/segmentation/best_model.pth'
        )
        return analyzer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def main():
    """Main application"""
    
    # Header
    st.markdown("<h1>🏠 Floor Plan Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 18px; color: #1976d2;'>
                📐 Analyze floor plans with AI-powered room segmentation and furniture detection
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Create tabs in sidebar
        sidebar_tab = st.radio(
            "Navigation",
            ["⚙️ Settings", "🤖 AI Assistant"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if sidebar_tab == "⚙️ Settings":
            st.header("⚙️ Settings")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload Floor Plan Image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a floor plan image for analysis"
            )
            
            # Save file bytes to session state
            if uploaded_file is not None:
                st.session_state.uploaded_file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.has_file = True
                uploaded_file.seek(0)  # Reset file pointer
            elif 'has_file' not in st.session_state:
                st.session_state.has_file = False
            
            st.markdown("---")
            
            # Calibration settings
            st.subheader("📏 Scale Calibration")
            
            ref_pixels = st.number_input(
                "Reference Length (pixels)",
                min_value=10,
                max_value=1000,
                value=200,
                help="Length of a known distance in the image (in pixels)"
            )
            
            ref_length_cm = st.number_input(
                "Actual Length (cm)",
                min_value=10.0,
                max_value=1000.0,
                value=200.0,
                help="Actual length of the reference in centimeters"
            )
            
            # Save to session state
            st.session_state.ref_pixels = ref_pixels
            st.session_state.ref_length_cm = ref_length_cm
            
            st.info(f"📊 Scale: {ref_pixels/ref_length_cm:.4f} pixels/cm")
            
            st.markdown("---")
            
            # About
            with st.expander("ℹ️ About"):
                st.markdown("""
                **Floor Plan Analyzer** uses deep learning to:
                - 🎯 Detect furniture and fixtures
                - 🏗️ Segment rooms and walls
                - 📐 Calculate room areas
                
                **Models:**
                - Detection: YOLOv8
                - Segmentation: U-Net (ResNet34)
                """)
        
        else:  # AI Assistant tab
            st.header("🤖 AI Assistant")
            st.caption("Powered by Ollama")
            
            # Initialize assistant first (before reading installed models)
            if 'ollama_assistant' not in st.session_state:
                st.session_state.ollama_assistant = OllamaAssistant()
            
            if 'ai_chat' not in st.session_state:
                st.session_state.ai_chat = []
            
            # Model selector — prefer installed models; fall back to common names
            installed = st.session_state.ollama_assistant.list_installed_models()
            model_options = installed if installed else ["gemma3", "llama3.2", "llama3.1", "mistral", "qwen2.5"]
            default_idx = 0
            if 'gemma3' in model_options:
                default_idx = model_options.index('gemma3')
            elif st.session_state.ollama_assistant.model in model_options:
                default_idx = model_options.index(st.session_state.ollama_assistant.model)
            model_name = st.selectbox(
                "Select Model:",
                model_options,
                index=default_idx,
                help="Only models you have pulled with `ollama pull` appear here"
            )
            st.session_state.ollama_assistant.model = model_name
            
            # Check status
            ollama_status = st.session_state.ollama_assistant.check_status()
            
            if ollama_status:
                st.success("✅ Ollama is running")
            else:
                st.error("❌ Ollama is not running")
                st.markdown(f"""
                **Setup Instructions:**
                1. Open terminal/command prompt
                2. Download model: `ollama pull {model_name}`
                3. Ollama will start automatically
                4. Refresh this page
                
                **Popular models:**
                - `ollama pull llama3.2` (recommended, ~2GB)
                - `ollama pull mistral` (fast, ~4GB)
                - `ollama pull qwen2.5` (Chinese support)
                """)
                st.stop()
            
            st.markdown("---")
            
            st.markdown("---")
            
            # Check if analysis is done
            has_analysis = 'analyzed' in st.session_state and st.session_state.analyzed
            
            if has_analysis:
                results = st.session_state.results
                context = st.session_state.ollama_assistant.create_context(results)
                
                rooms = results['rooms']
                furniture = results['furniture']
                total_area = sum(r['area_m2'] for r in rooms)
                
                st.markdown(f"""
                **📊 Analysis Summary:**
                - 🏠 Rooms: {len(rooms)}
                - 📐 Total Area: {total_area:.2f} m²
                - 🪑 Furniture: {len(furniture)} items
                """)
            else:
                st.warning("⚠️ No analysis data yet. Upload and analyze a floor plan first!")
                context = None
            
            st.markdown("---")
            
            # Welcome message
            if len(st.session_state.ai_chat) == 0:
                if has_analysis:
                    welcome = "👋 Hello! I've analyzed your floor plan. Ask me anything!"
                else:
                    welcome = "👋 Hello! Upload and analyze a floor plan, then I can help you with design insights!"
                st.session_state.ai_chat.append({'role': 'assistant', 'content': welcome})
            
            # Display chat history
            st.markdown("**💬 Chat:**")
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.ai_chat:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
            
            # Quick action buttons (only if analysis is done)
            if has_analysis:
                st.markdown("---")
                st.markdown("**⚡ Quick Actions:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💡 Get Analysis", key="quick_analysis", use_container_width=True):
                        prompt = "Please analyze this floor plan and provide professional insights."
                        st.session_state.ai_chat.append({'role': 'user', 'content': prompt})
                        with st.spinner("🤔 Thinking..."):
                            answer = st.session_state.ollama_assistant.chat(prompt, context)
                        st.session_state.ai_chat.append({'role': 'assistant', 'content': answer})
                        st.rerun()
                
                with col2:
                    if st.button("🔧 Suggestions", key="quick_suggest", use_container_width=True):
                        prompt = "What improvements do you recommend for this floor plan?"
                        st.session_state.ai_chat.append({'role': 'user', 'content': prompt})
                        with st.spinner("🤔 Thinking..."):
                            answer = st.session_state.ollama_assistant.chat(prompt, context)
                        st.session_state.ai_chat.append({'role': 'assistant', 'content': answer})
                        st.rerun()
                
                if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                    st.session_state.ai_chat = []
                    st.session_state.ollama_assistant.reset()
                    st.rerun()
            
            st.markdown("---")
            
            # Chat input box - ALWAYS visible
            user_input = st.chat_input("💬 Type your question here...")
            
            if user_input:
                if not has_analysis:
                    # No analysis yet
                    st.session_state.ai_chat.append({'role': 'user', 'content': user_input})
                    st.session_state.ai_chat.append({
                        'role': 'assistant', 
                        'content': "Please analyze a floor plan first! Go to Settings → Upload image → Click Analyze."
                    })
                    st.rerun()
                else:
                    # Add user message
                    st.session_state.ai_chat.append({'role': 'user', 'content': user_input})
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        answer = st.session_state.ollama_assistant.chat(user_input, context)
                    
                    # Add AI response
                    st.session_state.ai_chat.append({'role': 'assistant', 'content': answer})
                    st.rerun()
    
    # Main content
    # Get parameters from session state
    has_file = st.session_state.get('has_file', False)
    ref_pixels = st.session_state.get('ref_pixels', 200)
    ref_length_cm = st.session_state.get('ref_length_cm', 200.0)
    
    if has_file:
        # Create columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Uploaded Image")
            
            # Save uploaded file from session state
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(st.session_state.uploaded_file_bytes)
                tmp_path = tmp_file.name
            
            # Save to session state for later use
            st.session_state.tmp_path = tmp_path
            
            # Display original image
            image = Image.open(tmp_path)
            st.image(image, width='stretch')
        
        with col2:
            st.subheader("🔄 Analysis Status")
            status_placeholder = st.empty()
        
        # Analyze button
        if st.button("🚀 Analyze Floor Plan", type="primary"):
            with st.spinner("🔍 Analyzing floor plan..."):
                # Load analyzer
                analyzer = load_analyzer()
                
                if analyzer is None:
                    st.error("❌ Failed to load models. Please check model files.")
                    return
                
                # Read image for analysis
                image_cv = cv2.imread(tmp_path)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                # Update status
                status_placeholder.info("🏗️ Segmenting rooms...")
                
                # Analyze
                results = analyzer.analyze(
                    image_rgb,
                    tmp_path,
                    ref_pixels,
                    ref_length_cm
                )
                
                status_placeholder.success("✅ Analysis Complete!")
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.image_rgb = image_rgb
                st.session_state.analyzed = True
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🖼️ Visualization",
                    "🏠 Room Analysis",
                    "🪑 Furniture Detection",
                    "📈 Statistics"
                ])
                
                with tab1:
                    st.subheader("Room Segmentation Overlay")
                    overlay_rgb = cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB)
                    st.image(overlay_rgb, width='stretch')
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("Segmentation Mask")
                        st.image(results['colored_mask'], width='stretch')
                    with col_b:
                        st.caption("Legend")
                        st.markdown("""
                        - 🟢 **Green**: Room areas
                        - ⚫ **Gray**: Walls
                        - ⚫ **Black**: Background
                        """)
                
                with tab2:
                    st.subheader("Room Areas")
                    
                    rooms = results['rooms']
                    total_area = sum(r['area_m2'] for r in rooms)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rooms", len(rooms))
                    with col2:
                        st.metric("Total Area", f"{total_area:.2f} m²")
                    with col3:
                        if len(rooms) > 0:
                            st.metric("Largest Room", f"{rooms[0]['area_m2']:.2f} m²")
                    
                    st.markdown("---")
                    
                    # Room details table
                    st.subheader("Room Details")
                    
                    room_data = []
                    for i, room in enumerate(rooms, 1):
                        room_data.append({
                            'Room': f"Room {i}",
                            'Area (m²)': f"{room['area_m2']:.2f}",
                            'Area (pixels)': f"{room['area_pixels']:,}",
                            'Percentage': f"{(room['area_m2']/total_area)*100:.1f}%"
                        })
                    
                    st.dataframe(room_data, width='stretch')
                    
                    # Area chart
                    if len(rooms) > 0:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f"Room {i}" for i in range(1, min(len(rooms)+1, 11))],
                                y=[r['area_m2'] for r in rooms[:10]],
                                marker_color='lightblue',
                                text=[f"{r['area_m2']:.1f} m²" for r in rooms[:10]],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="Room Area Distribution (Top 10)",
                            xaxis_title="Room",
                            yaxis_title="Area (m²)",
                            height=400
                        )
                        st.plotly_chart(fig)
                
                with tab3:
                    st.subheader("Furniture Detection")
                    
                    furniture = results['furniture']
                    furniture_counts = Counter([f['class'] for f in furniture])
                    
                    # Summary metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Items", len(furniture))
                    with col2:
                        st.metric("Unique Types", len(furniture_counts))
                    
                    # Debug info - show what was detected
                    with st.expander("🔍 Debug: Detection Details"):
                        st.write("**Detected furniture types:**")
                        for ftype, count in furniture_counts.items():
                            st.write(f"- {ftype}: {count} item(s)")
                        st.write(f"\n**Total detections:** {len(furniture)}")
                        if len(furniture) > 0 and 'bbox' in furniture[0]:
                            st.success("✅ Bounding box data available")
                        else:
                            st.error("❌ Bounding box data missing")
                    
                    st.markdown("---")
                    
                    # Interactive furniture detection visualization
                    if len(furniture) > 0:
                        # Check if furniture items have bbox information
                        has_bbox = len(furniture) > 0 and 'bbox' in furniture[0]
                        
                        if has_bbox:
                            st.subheader("🎯 Interactive Detection View")
                            st.caption("Hover over the furniture items to highlight the corresponding detection boxes")
                            
                            # Create interactive Plotly figure
                            fig = go.Figure()
                            
                            # Add the base image
                            fig.add_layout_image(
                                dict(
                                    source=Image.fromarray(image_rgb),
                                    xref="x",
                                    yref="y",
                                    x=0,
                                    y=0,
                                    sizex=image_rgb.shape[1],
                                    sizey=image_rgb.shape[0],
                                    sizing="stretch",
                                    layer="below"
                                )
                            )
                            
                        # Color map for different furniture types
                        unique_classes = list(set([f['class'] for f in furniture]))
                        colors = px.colors.qualitative.Set1[:len(unique_classes)]
                        color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
                        
                        # Track which classes we've added to legend
                        legend_added = set()
                        
                        # Add bounding boxes as scatter traces (one trace per furniture item)
                        for idx, item in enumerate(furniture):
                            bbox = item['bbox']  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = bbox
                            
                            # Create rectangle coordinates
                            x_coords = [x1, x2, x2, x1, x1]
                            y_coords = [y1, y1, y2, y2, y1]
                            
                            furniture_type = item['class'].replace('_', ' ').title()
                            confidence = item['confidence']
                            furniture_class = item['class']
                            
                            # Only show legend for first item of each class
                            show_in_legend = furniture_class not in legend_added
                            if show_in_legend:
                                legend_added.add(furniture_class)
                            
                            # Add trace for each bounding box
                            fig.add_trace(go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='lines',
                                line=dict(color=color_map[furniture_class], width=3),
                                fill='toself',
                                fillcolor=color_map[furniture_class],
                                opacity=0.3,
                                name=furniture_type,  # Use class name for legend
                                legendgroup=furniture_class,
                                showlegend=show_in_legend,  # Only show once per class
                                visible=True,  # Ensure all are visible by default
                                hovertemplate=f"<b>{furniture_type} #{idx+1}</b><br>" +
                                            f"Confidence: {confidence:.2%}<br>" +
                                            f"Location: ({x1:.0f}, {y1:.0f})<br>" +
                                            f"Size: {x2-x1:.0f} × {y2-y1:.0f} px<br>" +
                                            "<extra></extra>"
                            ))
                        
                        # Update layout
                        fig.update_xaxes(
                            visible=False,
                            range=[0, image_rgb.shape[1]]
                        )
                        
                        fig.update_yaxes(
                            visible=False,
                            range=[image_rgb.shape[0], 0],  # Invert y-axis to match image coordinates
                            scaleanchor="x",
                            scaleratio=1
                        )
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=30, b=0),
                            hovermode='closest',
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(255, 255, 255, 0.9)",
                                bordercolor="Black",
                                borderwidth=1
                            ),
                            title=dict(
                                text="Furniture Detection - Click legend to show/hide types",
                                x=0.5,
                                xanchor='center'
                            )
                        )
                        
                        # Add instruction
                        st.info("💡 **Tip:** Click on legend items on the right to show/hide furniture types. Hover over boxes for details.")
                        
                        st.plotly_chart(fig)
                    else:
                        st.warning("⚠️ Please re-analyze the image to see the interactive detection view with bounding boxes.")
                    
                    st.markdown("---")
                    
                    # Furniture counts
                    st.subheader("Furniture Inventory")
                    
                    furniture_data = []
                    for item, count in sorted(furniture_counts.items()):
                        furniture_data.append({
                            'Item': item.replace('_', ' ').title(),
                            'Quantity': count
                        })
                    
                    st.dataframe(furniture_data, width='stretch')
                    
                    # Detailed furniture list
                    if has_bbox:
                        st.subheader("📋 Detailed Detection List")
                        for idx, item in enumerate(furniture, 1):
                            furniture_type = item['class'].replace('_', ' ').title()
                            confidence = item['confidence']
                            bbox = item['bbox']
                            
                            with st.expander(f"#{idx} - {furniture_type} (Confidence: {confidence:.2%})"):
                                col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Type:** {furniture_type}")
                        st.write(f"**Confidence:** {confidence:.2%}")
                    with col_b:
                        st.write(f"**Location:** ({bbox[0]:.0f}, {bbox[1]:.0f})")
                        st.write(f"**Size:** {bbox[2]-bbox[0]:.0f} × {bbox[3]-bbox[1]:.0f} px")
                    
                    # Pie chart
                    if len(furniture_counts) > 0:
                        st.subheader("📈 Furniture Distribution")
                        fig_pie = px.pie(
                            values=list(furniture_counts.values()),
                            names=[k.replace('_', ' ').title() for k in furniture_counts.keys()],
                            title="Furniture Distribution"
                        )
                        st.plotly_chart(fig_pie)
                
                with tab4:
                    st.subheader("Analysis Statistics")
                    
                    # Overall stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Area", f"{total_area:.2f} m²")
                    with col2:
                        st.metric("Rooms Detected", len(rooms))
                    with col3:
                        st.metric("Furniture Items", len(furniture))
                    with col4:
                        avg_room_area = total_area / len(rooms) if len(rooms) > 0 else 0
                        st.metric("Avg Room Size", f"{avg_room_area:.2f} m²")
                    
                    st.markdown("---")
                    
                    # Segmentation stats
                    mask = results['mask']
                    st.subheader("Segmentation Statistics")
                    
                    seg_stats = []
                    for class_id, class_name in analyzer.seg_class_names.items():
                        pixels = np.sum(mask == class_id)
                        percentage = (pixels / mask.size) * 100
                        seg_stats.append({
                            'Category': class_name,
                            'Pixels': f"{pixels:,}",
                            'Percentage': f"{percentage:.2f}%"
                        })
                    
                    st.dataframe(seg_stats, width='stretch')
                    
                    # Download button
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=f"""
FLOOR PLAN ANALYSIS REPORT
===========================

Total Area: {total_area:.2f} m²
Number of Rooms: {len(rooms)}
Number of Furniture Items: {len(furniture)}

ROOM DETAILS:
{chr(10).join([f"Room {i}: {r['area_m2']:.2f} m²" for i, r in enumerate(rooms, 1)])}

FURNITURE INVENTORY:
{chr(10).join([f"{item}: {count}" for item, count in furniture_counts.items()])}
                        """,
                        file_name="floor_plan_analysis.txt",
                        mime="text/plain"
                    )
    
    else:
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 60px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             border-radius: 20px; color: white;'>
            <h2>👋 Welcome to Floor Plan Analyzer!</h2>
            <p style='font-size: 18px; margin-top: 20px;'>
                Upload a floor plan image to get started with AI-powered analysis
            </p>
            <p style='font-size: 16px; margin-top: 15px;'>
                ⬅️ Use the sidebar to upload your image
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("---")
        st.header("✨ Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Object Detection
            Automatically detect and count:
            - Doors & Windows
            - Furniture items
            - Fixtures & Appliances
            """)
        
        with col2:
            st.markdown("""
            ### 🏗️ Room Segmentation
            Intelligent segmentation of:
            - Individual rooms
            - Walls & structures
            - Open spaces
            """)
        
        with col3:
            st.markdown("""
            ### 📐 Area Calculation
            Precise measurements:
            - Room dimensions
            - Total floor area
            - Percentage distribution
            """)


if __name__ == "__main__":
    main()

