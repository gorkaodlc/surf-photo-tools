import sys
from wave_splitter import app, open_browser_when_ready
import threading


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5070
    print(f"\n🌊 Wave Splitter → http://127.0.0.1:{port}\n")
    threading.Thread(target=open_browser_when_ready, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
