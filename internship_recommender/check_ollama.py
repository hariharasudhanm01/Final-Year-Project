"""
Quick script to check Ollama status and available models
Run this to diagnose chatbot connection issues
"""
import requests
import json

def check_ollama():
    base_url = "http://localhost:11434"
    
    print("=" * 60)
    print("Ollama Status Check")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama server is running on localhost:11434")
            print()
            
            # Get available models
            data = response.json()
            models = data.get('models', [])
            
            if models:
                print(f"Available models ({len(models)}):")
                print("-" * 60)
                for model in models:
                    model_name = model.get('name', 'Unknown')
                    model_size = model.get('size', 0)
                    size_gb = model_size / (1024**3) if model_size else 0
                    print(f"  - {model_name} ({size_gb:.2f} GB)")
                print()
                
                # Check for common model names
                model_names = [m.get('name', '') for m in models]
                if any('llama3.2' in name or 'llama3' in name for name in model_names):
                    print("[OK] Found Llama 3 model - chatbot should work!")
                elif any('llama' in name.lower() for name in model_names):
                    print("[WARNING] Found Llama model but not llama3.2")
                    print("  You may need to update the model name in hr_chatbot.py")
                else:
                    print("[WARNING] No Llama models found")
                    print("  Install with: ollama pull llama3.2")
            else:
                print("[WARNING] No models installed")
                print("  Install with: ollama pull llama3.2")
        else:
            print(f"[ERROR] Ollama responded with status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama server")
        print("  Error: Connection refused")
        print()
        print("To start Ollama:")
        print("  1. Open a new terminal")
        print("  2. Run: ollama serve")
        print("  3. Keep that terminal open")
    except Exception as e:
        print(f"[ERROR] Error checking Ollama: {e}")
    
    print()
    print("=" * 60)
    print("Test chatbot connection:")
    print("  The chatbot will automatically use available models")
    print("=" * 60)

if __name__ == "__main__":
    check_ollama()

