import ssl
import socket

def test_ssl_connection():
    try:
        context = ssl.create_default_context()
        with socket.create_connection(('query1.finance.yahoo.com', 443)) as sock:
            with context.wrap_socket(sock, server_hostname='query1.finance.yahoo.com') as ssock:
                print("✅ SSL 연결 성공")
                return True
    except Exception as e:
        print(f"❌ SSL 연결 실패: {e}")
        return False

if __name__ == "__main__":
    test_ssl_connection()
