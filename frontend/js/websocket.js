/**
 * Urban Acoustic Intelligence — Real-Time WebSocket Streaming Client
 */

export class AcousticWebSocketClient {
  constructor(endpointPath = '/ws/live', onEventCallback = null) {
    this.endpointPath = endpointPath;
    this.onEventCallback = onEventCallback;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectDelay = 10000;
    this.pingInterval = null;
    this.isConnected = false;

    this.connect();
  }

  getWebSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    return `${protocol}//${host}${this.endpointPath}`;
  }

  connect() {
    const url = this.getWebSocketUrl();
    console.log(`[WebSocket] Connecting to ${url}...`);
    this.updateStatusUI('connecting');

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('[WebSocket] Live stream connected.');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.updateStatusUI('connected');
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        if (event.data === 'pong') {
          // Heartbeat response
          return;
        }

        try {
          const payload = JSON.parse(event.data);
          if (this.onEventCallback) {
            this.onEventCallback(payload);
          }
        } catch (err) {
          console.warn('[WebSocket] Malformed JSON message:', event.data);
        }
      };

      this.ws.onclose = () => {
        console.warn('[WebSocket] Live stream disconnected. Reconnecting...');
        this.isConnected = false;
        this.stopHeartbeat();
        this.updateStatusUI('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error('[WebSocket] Error occurred:', err);
        this.ws.close();
      };
    } catch (err) {
      console.error('[WebSocket] Initialization error:', err);
      this.scheduleReconnect();
    }
  }

  startHeartbeat() {
    this.stopHeartbeat();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 20000);
  }

  stopHeartbeat() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), this.maxReconnectDelay);
    setTimeout(() => this.connect(), delay);
  }

  updateStatusUI(status) {
    const badgeEl = document.getElementById('ws-status-badge');
    const dotEl = document.getElementById('ws-pulse-dot');
    const textEl = document.getElementById('ws-status-text');

    if (!textEl || !dotEl) return;

    if (status === 'connected') {
      textEl.textContent = 'Live WebSocket';
      dotEl.className = 'pulse-dot';
      dotEl.style.backgroundColor = '#10b981';
    } else if (status === 'connecting') {
      textEl.textContent = 'Connecting...';
      dotEl.className = 'pulse-dot';
      dotEl.style.backgroundColor = '#f59e0b';
    } else {
      textEl.textContent = 'Offline';
      dotEl.className = 'pulse-dot disconnected';
      dotEl.style.backgroundColor = '#ef4444';
    }
  }
}
