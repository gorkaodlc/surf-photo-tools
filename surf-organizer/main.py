import sys
import webbrowser
import threading
from surf_organizer_web import app


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n🏄 Surf Photo Organizer → http://localhost:{port}\n")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
