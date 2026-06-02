import os
import sys
import time
import subprocess
import threading
import http.server
import socketserver
import urllib.request
import json
import asyncio
import base64

# Add the websockets package from virtual environment
# The script is executed from the repo virtual environment, so it will import it normally.
try:
    import websockets
except ImportError:
    print("Error: 'websockets' library is required. Please run using the repository virtual environment python.")
    sys.exit(1)

PORT = 8099
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PRESENTATION_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PRESENTATION_DIR, "VocaRig_Sunum.pdf")
global_httpd = None

class SilentHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress server request logging
        pass

def run_http_server():
    global global_httpd
    # Serve from presentation directory
    handler = lambda *args, **kwargs: SilentHTTPRequestHandler(*args, directory=PRESENTATION_DIR, **kwargs)
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            global_httpd = httpd
            httpd.serve_forever()
    except Exception as e:
        print(f"HTTP Server error: {e}")

async def generate_pdf():
    # Fetch WebSocket debugger URL
    ws_url = None
    for attempt in range(15):
        try:
            response = urllib.request.urlopen("http://localhost:9222/json/list", timeout=2)
            targets = json.loads(response.read().decode())
            for t in targets:
                if t.get('type') == 'page':
                    ws_url = t.get('webSocketDebuggerUrl')
                    break
            if ws_url:
                break
        except Exception:
            pass
        print("Waiting for Chrome debugger connection (attempt {}/15)...".format(attempt + 1))
        await asyncio.sleep(0.5)

    if not ws_url:
        # Try to open a new target page if none exists
        try:
            response = urllib.request.urlopen("http://localhost:9222/json/new", timeout=2)
            target = json.loads(response.read().decode())
            ws_url = target.get('webSocketDebuggerUrl')
        except Exception as e:
            print(f"Failed to create new page target: {e}")
            return False

    if not ws_url:
        print("Error: Could not retrieve WebSocket debugger URL from Chrome.")
        return False

    print(f"Connected to Chrome WebSocket: {ws_url}")
    
    async with websockets.connect(ws_url, max_size=50*1024*1024) as ws:
        # Navigate to index.html with light theme query parameter
        print("Navigating to presentation page in light theme...")
        navigate_cmd = {
            "id": 1,
            "method": "Page.navigate",
            "params": {
                "url": f"http://localhost:{PORT}/index.html?theme=light"
            }
        }
        await ws.send(json.dumps(navigate_cmd))
        await ws.recv()
        
        # Wait for layouts, fonts, and initialization to settle
        print("Waiting 4 seconds for animations and layout rendering to settle...")
        await asyncio.sleep(4.0)
        
        # Trigger print to PDF
        print("Generating PDF via Chrome DevTools Protocol...")
        print_cmd = {
            "id": 2,
            "method": "Page.printToPDF",
            "params": {
                "landscape": True,
                "printBackground": True,
                "preferCSSPageSize": True,
                "displayHeaderFooter": False,
                "marginPattern": 0  # No margins
            }
        }
        await ws.send(json.dumps(print_cmd))
        
        # Receive base64 response
        pdf_data = None
        while True:
            resp_str = await ws.recv()
            resp_json = json.loads(resp_str)
            if resp_json.get("id") == 2:
                if "error" in resp_json:
                    print(f"CDP print error: {resp_json['error']}")
                    return False
                pdf_data = resp_json["result"]["data"]
                break
        
        print("Decoding and writing PDF to file...")
        pdf_bytes = base64.b64decode(pdf_data)
        with open(PDF_PATH, "wb") as f:
            f.write(pdf_bytes)
        
        print(f"PDF exported successfully to: {PDF_PATH}")
        return True

def main():
    print("Starting background local HTTP server...")
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
    # Wait for HTTP server to bind
    time.sleep(1.0)
    
    print("Launching Google Chrome in headless debugging mode...")
    user_data_dir = os.path.join(PRESENTATION_DIR, ".chrome-profile")
    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1920,1080"
    ]
    
    chrome_proc = None
    try:
        chrome_proc = subprocess.Popen(chrome_cmd)
        
        # Run async PDF generator
        success = asyncio.run(generate_pdf())
        if not success:
            print("PDF generation failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)
        
    finally:
        print("Shutting down processes...")
        if chrome_proc:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                chrome_proc.kill()
        
        if global_httpd:
            global_httpd.shutdown()
        print("Done!")

if __name__ == "__main__":
    main()
