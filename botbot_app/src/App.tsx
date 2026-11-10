/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Send,
  Check,
  CheckCheck,
  Activity,
  Sparkles,
  Search,
  FileText,
  Menu,
  X,
  Upload,
  Globe,
  Database,
  Settings,
  ArrowLeft,
  CornerUpLeft,
  Palette,
  Copy,
  Plus,
  Brain,
  Camera,
  Bell
} from "lucide-react";
import { formatBytes, extractTextFromPDF } from "./lib/fileParser";
import { ChatMessage, WallpaperConfig, Attachment } from "./types";
import AIEngineInspector from "./components/AIEngineInspector";
import WallpaperSettings from "./components/WallpaperSettings";
import { LocalNotifications } from '@capacitor/local-notifications';
import { App as CapApp } from '@capacitor/app';
import { Clipboard } from '@capacitor/clipboard';
import { toPng } from 'html-to-image';

// --- CoT diagnostic mode (separate one-shot REST path; NOT the websocket pipeline) ---
interface CoTEntry {
  id: string;
  prompt: string;
  timestamp: number;
  status: "loading" | "ok" | "parse_failed" | "error";
  reasoning?: { content: string; note?: string };
  reply?: string;
  errorDetail?: string;
  rawOutput?: string;
  measuredTokens?: number;
  durationS?: number;
}

// Backend allows ~900s of generation on this endpoint; client timeout sits above it.
const COT_TIMEOUT_MS = 950000;

// --- IndexedDB Offline Queue Helper ---
export interface QueuedMessage {
  id: string;
  session_id: string;
  text: string;
  created_at: number;
  status: "pending" | "sent" | "acked";
  attachments?: any[];
  replyTo?: {
    id: string;
    sender: "user" | "bot";
    text: string;
  };
}

const DB_NAME = "botbot_chat_db";
const STORE_NAME = "offline_queue";
const DB_VERSION = 1;

function openQueueDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("session_id", "session_id", { unique: false });
        store.createIndex("created_at", "created_at", { unique: false });
        store.createIndex("status", "status", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function enqueueDBMessage(msg: QueuedMessage): Promise<void> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(msg);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getPendingDBMessages(sessionId: string): Promise<QueuedMessage[]> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index("session_id");
    const request = index.getAll(IDBKeyRange.only(sessionId));
    request.onsuccess = () => {
      const messages = (request.result as QueuedMessage[])
        .filter((m) => m.status !== "acked")
        .sort((a, b) => a.created_at - b.created_at);
      resolve(messages);
    };
    request.onerror = () => reject(request.error);
  });
}

async function updateDBMessageStatus(id: string, status: "pending" | "sent" | "acked"): Promise<void> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      if (getReq.result) {
        store.put({ ...getReq.result, status });
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function removeAckedDBMessage(id: string): Promise<void> {
  const db = await openQueueDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

interface BotIdentityConfig {
  name: string;
  nickname: string;
  avatar: string;
}

const DEFAULT_BOT_CONFIG: BotIdentityConfig = {
  name: "BotBot AI",
  nickname: "BotBot",
  avatar: "🤖"
};

const INITIAL_MESSAGES: ChatMessage[] = [];

function safeGetItem(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch (e) {
    console.warn(`[storage] Failed to get ${key}:`, e);
    return null;
  }
}

function safeSetItem(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch (e) {
    console.warn(`[storage] Failed to set ${key} (quota exceeded?):`, e);
  }
}

function sanitizeMessagesForStorage(msgs: ChatMessage[]): any[] {
  // Cap at last 100 messages and strip bulky base64 data to protect localStorage quota
  return msgs.slice(-100).map((m) => {
    if (!m.attachments || m.attachments.length === 0) return m;
    const sanitizedAtts = m.attachments.map((att: any) => {
      const { base64, url, ...rest } = att;
      return rest;
    });
    return { ...m, attachments: sanitizedAtts };
  });
}

function saveMessagesToStorage(msgs: ChatMessage[]) {
  safeSetItem("whatsapp_single_bot_messages", JSON.stringify(sanitizeMessagesForStorage(msgs)));
}

export default function App() {
  const [botConfig, setBotConfig] = useState<BotIdentityConfig>(() => {
    const saved = safeGetItem("whatsapp_single_bot_config");
    if (!saved) return DEFAULT_BOT_CONFIG;
    try {
      return JSON.parse(saved);
    } catch {
      return DEFAULT_BOT_CONFIG;
    }
  });

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = safeGetItem("whatsapp_single_bot_messages");
    if (!saved) return INITIAL_MESSAGES;
    try {
      return JSON.parse(saved);
    } catch {
      return INITIAL_MESSAGES;
    }
  });

  const [fastApiBaseUrl, setFastApiBaseUrl] = useState<string>(() => {
    return safeGetItem("whatsapp_fastapi_base_url") || (import.meta as any).env?.VITE_FASTAPI_BASE_URL || "";
  });

  const [fastApiKey, setFastApiKey] = useState<string>(() => {
    return safeGetItem("whatsapp_fastapi_key") || (import.meta as any).env?.VITE_API_KEY || "";
  });

  const [sessionId, setSessionId] = useState<string>(() => {
    const saved = safeGetItem("whatsapp_fastapi_session_id");
    if (saved) return saved;
    const newId = "session-" + Math.random().toString(36).substring(2, 11);
    safeSetItem("whatsapp_fastapi_session_id", newId);
    return newId;
  });

  const [previousSessions, setPreviousSessions] = useState<string[]>(() => {
    const saved = safeGetItem("whatsapp_fastapi_previous_sessions");
    if (!saved) return [];
    try {
      return JSON.parse(saved);
    } catch {
      return [];
    }
  });

  useEffect(() => {
    if (sessionId) {
      setPreviousSessions(prev => {
        if (prev.includes(sessionId)) return prev;
        const next = [sessionId, ...prev].slice(0, 10);
        safeSetItem("whatsapp_fastapi_previous_sessions", JSON.stringify(next));
        return next;
      });
    }
  }, [sessionId]);

  const switchSession = (id: string) => {
    const trimmed = id.trim();
    if (!trimmed || trimmed === sessionId) return;
    setSessionId(trimmed);
    safeSetItem("whatsapp_fastapi_session_id", trimmed);
    setMessages([]);
    setCoTEntries([]);
    setIsTyping(false);
    addLog(`Switched to session: ${trimmed}`, "info");
    socketRef.current?.close();
  };

  const [wallpaper, setWallpaper] = useState<WallpaperConfig>(() => {
    const saved = safeGetItem("whatsapp_ai_wallpaper");
    if (!saved) return { type: "preset", value: "#efeae2" };
    try {
      return JSON.parse(saved);
    } catch {
      return { type: "preset", value: "#efeae2" };
    }
  });

  const [showPatterns, setShowPatterns] = useState(() => {
    const saved = safeGetItem("whatsapp_ai_show_patterns");
    if (!saved) return true;
    try {
      return JSON.parse(saved);
    } catch {
      return true;
    }
  });

  const [searchQuery, setSearchQuery] = useState("");
  const [showSearchBar, setShowSearchBar] = useState(false);
  const [showWallpaperDrawer, setShowWallpaperDrawer] = useState(false);
  const [showChatWallpaperDrawer, setShowChatWallpaperDrawer] = useState(false);
  const [replyingTo, setReplyingTo] = useState<ChatMessage | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [showInspectorPanel, setShowInspectorPanel] = useState(true);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const [attachedFiles, setAttachedFiles] = useState<Attachment[]>([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // CoT diagnostic mode — entries are persisted locally for transcript
  // continuity across restarts; they are never sent to the offline queue and
  // never stored server-side (the backend does not save CoT turns).
  const [cotMode, setCotMode] = useState<boolean>(() => {
    const saved = safeGetItem("whatsapp_cot_mode");
    if (!saved) return false;
    try {
      return JSON.parse(saved);
    } catch {
      return false;
    }
  });
  const [cotEntries, setCoTEntries] = useState<CoTEntry[]>(() => {
    const saved = safeGetItem("whatsapp_cot_entries");
    if (!saved) return [];
    let parsed: any;
    try {
      parsed = JSON.parse(saved);
    } catch {
      return [];
    }
    if (!Array.isArray(parsed)) return [];
    // An entry persisted while a CoT request was in flight restores as
    // "loading" with no live fetch behind it (the request died with the
    // previous app process) — resolve it to a visible error instead of a
    // "CoT generating..." bubble that could never clear.
    return parsed
      .slice(-30)
      .map((entry: CoTEntry) =>
        entry && entry.status === "loading"
          ? {
            ...entry,
            status: "error" as const,
            errorDetail: "The app closed before this CoT response arrived — the request is no longer running.",
          }
          : entry
      );
  });
  const [isCoTLoading, setIsCoTLoading] = useState(false);

  const [logs, setLogs] = useState<Array<{ time: string; type: "info" | "api" | "error"; text: string }>>([]);
  const [activePayload, setActivePayload] = useState<any>(null);

  const [userName, setUserName] = useState(() => {
    return safeGetItem("whatsapp_ai_username") || "WiFi Client Client";
  });
  const [isEditingName, setIsEditingName] = useState(false);

  const [viewingDocument, setViewingDocument] = useState<Attachment | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionTestResult, setConnectionTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatFlowRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sessionInputRef = useRef<HTMLInputElement>(null);
  const toastTimeoutRef = useRef<any>(null);

  const showToast = useCallback((msg: string) => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setToastMessage(msg);
    toastTimeoutRef.current = setTimeout(() => {
      setToastMessage(null);
    }, 2500);
  }, []);

  const copyToClipboard = useCallback(async (text: string) => {
    let copied = false;
    try {
      await Clipboard.write({ string: text });
      copied = true;
    } catch {
      // Ignore Capacitor clipboard error
    }
    if (!copied && typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch {
        // Ignore navigator clipboard error
      }
    }
    if (!copied) {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.left = "-999999px";
        textarea.style.top = "-999999px";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const res = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (res) copied = true;
      } catch (e) {
        console.warn("execCommand copy failed", e);
      }
    }
    if (copied) {
      showToast("Copied to clipboard!");
    } else {
      showToast("Failed to copy");
    }
  }, [showToast]);

  const triggerNotification = useCallback(async (title: string, body: string) => {
    let dispatched = false;
    if (typeof window !== "undefined" && "Notification" in window) {
      if (Notification.permission === "granted") {
        try {
          new Notification(title, {
            body,
            icon: "/pwa-192x192.png"
          });
          dispatched = true;
        } catch (e) {
          console.debug("Web Notification failed", e);
        }
      }
    }
    try {
      await LocalNotifications.schedule({
        notifications: [{
          title,
          body,
          id: Math.floor(Math.random() * 1000000),
          schedule: { at: new Date(Date.now() + 100) },
          sound: 'default'
        }]
      });
      dispatched = true;
    } catch (notifyErr) {
      console.debug("LocalNotifications schedule failed", notifyErr);
    }
    return dispatched;
  }, []);

  const requestNotificationPermissions = useCallback(async (): Promise<boolean> => {
    let granted = false;
    if (typeof window !== "undefined" && "Notification" in window) {
      try {
        const perm = await Notification.requestPermission();
        if (perm === "granted") granted = true;
      } catch (e) {
        console.warn("Notification.requestPermission error", e);
      }
    }
    try {
      const status = await LocalNotifications.checkPermissions();
      if (status.display !== 'granted') {
        const req = await LocalNotifications.requestPermissions();
        if (req.display === 'granted') granted = true;
      } else {
        granted = true;
      }
    } catch (e) {
      console.debug("Capacitor LocalNotifications permission check failed", e);
    }
    return granted;
  }, []);

  const captureLongScreenshot = useCallback(async () => {
    if (!chatFlowRef.current) return;
    showToast("Generating screenshot...");
    try {
      const node = chatFlowRef.current;
      const dataUrl = await toPng(node, {
        backgroundColor: wallpaper.type === "solid" || wallpaper.type === "preset" ? wallpaper.value : "#efeae2",
        pixelRatio: 2,
        height: node.scrollHeight,
        style: {
          height: 'auto',
          maxHeight: 'none',
          overflow: 'visible'
        }
      });
      const link = document.createElement('a');
      link.download = `botbot_chat_${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
      showToast("Screenshot captured!");
    } catch (err) {
      console.error("Screenshot capture failed:", err);
      showToast("Screenshot capture failed");
    }
  }, [wallpaper, showToast]);

  const reconnectTimeoutRef = useRef<any>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const isFlushingRef = useRef<boolean>(false);
  const pendingAcksRef = useRef<Map<string, () => void>>(new Map());
  const lastAttemptRef = useRef<Map<string, number>>(new Map());
  const lastPongRef = useRef<number>(0);

  const addLog = (text: string, type: "info" | "api" | "error" = "info") => {
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    setLogs((prev) => [{ time, type, text }, ...prev.slice(0, 49)]);
  };

  const RETRY_COOLDOWN_MS = 10000; // longer than the 8s ack timeout below

  const flushQueue = useCallback(async () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN || isFlushingRef.current) {
      return;
    }

    isFlushingRef.current = true;
    try {
      let pending = await getPendingDBMessages(sessionId);
      while (pending.length > 0 && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        let sentAny = false;
        for (const item of pending) {
          if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) break;

          const sanitizedAttachments = item.attachments?.map(({ url, ...rest }: any) => rest);
          const payload = {
            type: "chat_message",
            id: item.id,
            session_id: item.session_id,
            text: item.text,
            message: item.text,
            attachments: sanitizedAttachments,
            reply_to: item.replyTo,
            created_at: item.created_at,
          };
          const payloadStr = JSON.stringify(payload);

          // Permanent failure guard: drop payloads > 10MB to prevent infinite crash loops
          if (payloadStr.length > 10 * 1024 * 1024) {
            await removeAckedDBMessage(item.id);
            setMessages((prev) => {
              const updated = prev.map((m) =>
                m.id === item.id
                  ? { ...m, status: "error" as const, text: (m.text || "") + " [Failed: attachment payload exceeds 10MB limit]" }
                  : m
              );
              saveMessagesToStorage(updated);
              return updated;
            });
            addLog(`Message ${item.id} dropped: attachment payload exceeds 10MB limit.`, "error");
            continue;
          }

          const lastAttempt = lastAttemptRef.current.get(item.id) ?? 0;
          if (Date.now() - lastAttempt < RETRY_COOLDOWN_MS) {
            continue;
          }
          lastAttemptRef.current.set(item.id, Date.now());
          sentAny = true;

          await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
              pendingAcksRef.current.delete(item.id);
              reject(new Error("Ack timeout (8s)"));
            }, 8000);

            pendingAcksRef.current.set(item.id, () => {
              clearTimeout(timeout);
              lastAttemptRef.current.delete(item.id);
              resolve();
            });

            socketRef.current?.send(payloadStr);
            updateDBMessageStatus(item.id, "sent");
            setMessages((prev) => {
              const updated = prev.map((m) => m.id === item.id ? { ...m, status: "delivered" as const } : m);
              saveMessagesToStorage(updated);
              return updated;
            });
          }).catch((err) => {
            setIsTyping(false);
            addLog(`Delivery retry queued for msg ${item.id}: ${err.message}`, "info");
          });
        }
        if (!sentAny) break;
        pending = await getPendingDBMessages(sessionId);
      }
    } finally {
      isFlushingRef.current = false;
    }
  }, [sessionId]);

  useEffect(() => {
    if (!fastApiBaseUrl) return;

    let heartbeatInterval: any;
    let cancelled = false;

    const scheduleReconnect = () => {
      if (cancelled) return;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);

      const delay = Math.min(1000 * 2 ** reconnectAttemptsRef.current, 30000);
      reconnectAttemptsRef.current += 1;
      addLog(`WebSocket reconnecting in ${delay / 1000}s...`, "info");

      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, delay);
    };

    const connectWebSocket = async () => {
      if (cancelled) return;

      if (!fastApiBaseUrl || !fastApiBaseUrl.startsWith("http")) {
        addLog("Waiting for valid Base URL in settings...", "info");
        return;
      }

      try {
        const base = fastApiBaseUrl.trim().replace(/\/+$/, "");

        const ticketRes = await fetch(`${base}/ws/ticket`, {
          method: "POST",
          headers: {
            "X-API-Key": fastApiKey,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ session_id: sessionId }),
        });

        if (!ticketRes.ok) {
          throw new Error(`Failed to get WebSocket ticket: ${ticketRes.status}`);
        }

        const { ticket } = await ticketRes.json();
        if (cancelled) return;

        const wsUrl = base.replace(/^https:\/\//i, "wss://").replace(/^http:\/\//i, "ws://") + `/ws/chat?ticket=${ticket}&session_id=${sessionId}`;
        addLog(`Connecting: ${wsUrl}`, "info");

        const socket = new WebSocket(wsUrl);
        socketRef.current = socket;

        socket.onopen = () => {
          if (cancelled) {
            socket.close();
            return;
          }
          addLog("Connected! (Background Service Active)", "info");
          setIsConnected(true);
          reconnectAttemptsRef.current = 0;

          lastPongRef.current = Date.now();

          heartbeatInterval = setInterval(() => {
            if (socket.readyState !== WebSocket.OPEN) return;

            const sinceLastPong = Date.now() - lastPongRef.current;
            if (sinceLastPong > 35000) {
              // Two missed pongs (15s interval, generous margin) on a socket
              // that still reports OPEN — it's half-open. Force a real
              // reconnect rather than waiting on this dead handle.
              addLog("No pong in 35s, socket appears half-open. Forcing reconnect.", "error");
              socket.close();
              return;
            }

            socket.send(JSON.stringify({ type: "ping" }));
          }, 15000);

          flushQueue();
        };

        socket.onmessage = async (event) => {
          let data: any;
          try {
            data = JSON.parse(event.data);
          } catch (parseErr) {
            addLog(`Received non-JSON WebSocket message, ignoring.`, "error");
            return;
          }

          if (data.type === "pong") {
            lastPongRef.current = Date.now();
            return;
          }

          // 1. Server ACK for user message
          if (data.type === "ack" && data.id) {
            await removeAckedDBMessage(data.id);
            setMessages((prev) => {
              const updated = prev.map((m) => m.id === data.id ? { ...m, status: "read" as const } : m);
              saveMessagesToStorage(updated);
              return updated;
            });
            const resolveAck = pendingAcksRef.current.get(data.id);
            if (resolveAck) {
              resolveAck();
              pendingAcksRef.current.delete(data.id);
            }
            return;
          }

          // 1b. Server confirmation the message is durably saved — arrives
          // later than "ack" (after generate_reply/save_turn completes).
          // Purely additive: doesn't touch the ack-driven retry/flush path,
          // doesn't gate anything time-sensitive. Safe no-op if unused.
          if (data.type === "stored" && data.id) {
            addLog(`Message ${data.id} durably stored on server.`, "info");
            return;
          }

          // 2. Incoming bot message
          const replyText = data.reply ?? data.text ?? data.message;
          if (!replyText) return;

          const botReplyId = data.id || "msg-ws-" + Math.random().toString(36).substring(2, 11);

          // Acknowledge receipt back to server
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ack", id: botReplyId }));
          }

          const botMsg: ChatMessage = {
            id: botReplyId,
            sender: "bot",
            text: replyText,
            timestamp: data.created_at || Date.now(),
            status: "read"
          };

          setIsTyping(false);
          setMessages((prev) => {
            if (prev.some((m) => m.id === botReplyId)) return prev;
            const updated = [...prev, botMsg];
            saveMessagesToStorage(updated);
            return updated;
          });

          // Trigger notification (Web Notification or Capacitor LocalNotifications)
          triggerNotification(botConfig.name, botMsg.text);
        };

        socket.onclose = () => {
          setIsConnected(false);
          setIsTyping(false);
          clearInterval(heartbeatInterval);
          if (socketRef.current === socket) {
            socketRef.current = null;
          }
          if (!cancelled) {
            scheduleReconnect();
          }
        };

        socket.onerror = () => {
          setIsTyping(false);
          addLog(`Socket connection dropped.`, "error");
          socket.close();
        };
      } catch (err: any) {
        // Includes the ticket endpoint returning non-200 (e.g. 503 while the
        // backend is down) — no reply can arrive on this connection.
        setIsTyping(false);
        addLog(`Connection attempt failed: ${err.message}`, "error");
        setIsConnected(false);
        if (!cancelled) {
          scheduleReconnect();
        }
      }
    };

    connectWebSocket();

    return () => {
      cancelled = true;
      clearInterval(heartbeatInterval);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [fastApiBaseUrl, sessionId, flushQueue]);

  useEffect(() => {
    let removeListener: (() => void) | undefined;
    let isMounted = true;

    CapApp.addListener('appStateChange', ({ isActive }) => {
      if (!isActive) {
        addLog("App backgrounded. Sleeping UI to save battery.", "info");
      } else {
        addLog("App resumed. Waking up UI.", "info");
        flushQueue();
      }
    }).then((handle) => {
      if (!isMounted) {
        handle.remove();
        return;
      }
      removeListener = () => handle.remove();
    });

    const backListener = CapApp.addListener('backButton', () => {
      if (showSettings) {
        setShowSettings(false);
      } else if (!isMobileSidebarOpen) {
        setIsMobileSidebarOpen(true);
      } else {
        CapApp.exitApp();
      }
    });

    return () => {
      isMounted = false;
      removeListener?.();
      backListener.then(h => h.remove());
    };
  }, [showSettings, isMobileSidebarOpen, flushQueue]);

  useEffect(() => {
    const requestPermissions = async () => {
      try {
        const status = await LocalNotifications.checkPermissions();
        if (status.display !== 'granted') {
          await LocalNotifications.requestPermissions();
        }
      } catch {
        // Ignored when running outside Capacitor mobile shell
      }
    };
    requestPermissions();
  }, []);

  useEffect(() => {
    safeSetItem("whatsapp_single_bot_config", JSON.stringify(botConfig));
  }, [botConfig]);

  useEffect(() => {
    saveMessagesToStorage(messages);
  }, [messages]);

  useEffect(() => {
    safeSetItem("whatsapp_fastapi_base_url", fastApiBaseUrl);
  }, [fastApiBaseUrl]);

  useEffect(() => {
    safeSetItem("whatsapp_fastapi_key", fastApiKey);
  }, [fastApiKey]);

  useEffect(() => {
    safeSetItem("whatsapp_ai_wallpaper", JSON.stringify(wallpaper));
  }, [wallpaper]);

  useEffect(() => {
    safeSetItem("whatsapp_ai_show_patterns", JSON.stringify(showPatterns));
  }, [showPatterns]);

  useEffect(() => {
    safeSetItem("whatsapp_cot_mode", JSON.stringify(cotMode));
  }, [cotMode]);

  useEffect(() => {
    safeSetItem("whatsapp_cot_entries", JSON.stringify(cotEntries.slice(-30)));
  }, [cotEntries]);

  useEffect(() => {
    safeSetItem("whatsapp_ai_username", userName);
  }, [userName]);

  useEffect(() => {
    if (!isMobileSidebarOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, isTyping, isMobileSidebarOpen, cotEntries]);

  const getNormalizedUrl = (baseUrl: string, endpoint: string) => {
    const base = baseUrl.trim().replace(/\/+$/, '');
    const path = endpoint.trim().replace(/^\/+/, '');
    return path ? `${base}/${path}` : base + "/";
  };

  const renderAvatar = (avatarStr: string, className = "w-10 h-10") => {
    if (avatarStr.startsWith("data:") || avatarStr.startsWith("http") || avatarStr.startsWith("blob:")) {
      return (
        <img
          src={avatarStr}
          alt="Bot Avatar"
          className={`${className} rounded-full object-cover shadow-inner border border-slate-200/50`}
          referrerPolicy="no-referrer"
        />
      );
    }
    return (
      <div className={`${className} rounded-full bg-emerald-50 border border-slate-200/50 flex items-center justify-center font-bold shadow-inner`}>
        <span className={className.includes("w-14") || className.includes("w-24") ? "text-2xl" : "text-base"}>
          {avatarStr || "🤖"}
        </span>
      </div>
    );
  };

  const handleAvatarUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      addLog("Failed to change avatar: selected file is not an image.", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result && typeof event.target.result === "string") {
        setBotConfig((prev) => ({ ...prev, avatar: event.target!.result as string }));
        addLog("Successfully updated bot profile picture!", "info");
      }
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file: File) => {
      const isImage = file.type.startsWith("image/");
      const isPdf = file.type === "application/pdf" || file.name.endsWith(".pdf");
      const isDocx = file.name.endsWith(".docx");
      const type = isImage ? "image" : isPdf ? "pdf" : isDocx ? "docx" : "text";

      const url = URL.createObjectURL(file);
      const reader = new FileReader();
      reader.onload = async (event) => {
        const result = event.target?.result as string;

        let textContents = undefined;
        if (type === "text") {
          try {
            textContents = await file.text();
          } catch {
            textContents = `[Text file attached: ${file.name}]`;
          }
        } else if (type === "pdf") {
          const arrayBuffer = await file.arrayBuffer();
          textContents = await Promise.resolve(extractTextFromPDF(arrayBuffer, file.name));
        } else if (type === "docx") {
          textContents = `[Word document attached: ${file.name} (${formatBytes(file.size)}) - text extraction is handled server-side]`;
        }

        const newAttachment: Attachment = {
          id: "att-" + Math.random().toString(36).substring(2, 11),
          name: file.name,
          type: type,
          size: formatBytes(file.size),
          url: url,
          base64: result,
          textContents: textContents
        };

        setAttachedFiles((prev) => [...prev, newAttachment]);
        addLog(`Attached ${type} file: "${file.name}"`, "info");
      };

      // Always read as Data URL so base64 is uniformly populated for all attachment types
      reader.readAsDataURL(file);
    });

    e.target.value = "";
  };

  const testLocalConnection = async (): Promise<boolean> => {
    setIsTestingConnection(true);
    setConnectionTestResult(null);
    const targetUrl = getNormalizedUrl(fastApiBaseUrl, "/");
    addLog(`Pinging backend: ${targetUrl}`, "info");

    try {
      const response = await fetch(targetUrl, {
        method: "GET",
        mode: "cors",
        headers: {
          "X-API-Key": fastApiKey
        }
      });
      setIsTestingConnection(false);

      if (response.ok) {
        const msg = `Connected to ${targetUrl} (Status: ${response.status})`;
        addLog(msg, "info");
        setConnectionTestResult({ success: true, message: msg });
        return true;
      } else {
        const msg = `Server found at ${targetUrl} but returned ${response.status}.`;
        addLog(msg, "error");
        setConnectionTestResult({ success: false, message: msg });
        return false;
      }
    } catch (err: any) {
      setIsTestingConnection(false);
      const errMsg = `Connection Failed: ${targetUrl}. (Error: ${err.message})`;
      addLog(errMsg, "error");
      setConnectionTestResult({ success: false, message: errMsg });
      return false;
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputText.trim() && attachedFiles.length === 0) return;

    const currentText = inputText.trim();
    const currentReplyTo = replyingTo;
    const currentAttachedFiles = [...attachedFiles];

    setInputText("");
    setAttachedFiles([]);
    setReplyingTo(null);

    const userMessageId = "msg-" + Math.random().toString(36).substring(2, 11);
    const userMessage: ChatMessage = {
      id: userMessageId,
      sender: "user",
      text: currentText,
      timestamp: Date.now(),
      status: "sent",
      attachments: currentAttachedFiles.length > 0 ? currentAttachedFiles : undefined,
      replyTo: currentReplyTo ? {
        id: currentReplyTo.id,
        sender: currentReplyTo.sender,
        text: currentReplyTo.text
      } : undefined
    };

    setMessages((prev) => {
      const updated = [...prev, userMessage];
      saveMessagesToStorage(updated);
      return updated;
    });

    setIsTyping(true);

    const queuedItem: QueuedMessage = {
      id: userMessageId,
      session_id: sessionId,
      text: currentText,
      created_at: userMessage.timestamp,
      status: "pending",
      attachments: currentAttachedFiles,
      replyTo: userMessage.replyTo
    };
    await enqueueDBMessage(queuedItem);

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      flushQueue();
    } else {
      // Message is only queued locally — nothing can generate a reply yet,
      // so don't leave the typing indicator running for an unsent message.
      setIsTyping(false);
      addLog("Offline: message buffered in storage.", "info");
    }
  };

  const handleNewChat = () => {
    const newId = "session-" + Math.random().toString(36).substring(2, 11);
    setSessionId(newId);
    safeSetItem("whatsapp_fastapi_session_id", newId);
    setMessages([]);
    saveMessagesToStorage([]);
    setCoTEntries([]);
    setIsTyping(false);
    addLog(`Started new chat session: ${newId}`, "info");
    socketRef.current?.close();
  };

  const updateCoTEntry = (id: string, patch: Partial<CoTEntry>) => {
    setCoTEntries((prev) => prev.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)));
  };

  // CoT diagnostic send — a separate one-shot REST call. Deliberately does NOT
  // touch the websocket, ack/dedupe, offline queue, or the isTyping indicator
  // (Fix B) — those belong to the normal chat pipeline only.
  const handleSendCoT = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const text = inputText.trim();
    if (!text) return;

    if (attachedFiles.length > 0) {
      setCoTEntries((prev) => [...prev, {
        id: "cot-" + Math.random().toString(36).substring(2, 11),
        prompt: text,
        timestamp: Date.now(),
        status: "error",
        errorDetail: "Attachments are not supported in CoT diagnostic mode (the /chat/cot endpoint accepts text only). Remove the attachment and send again.",
      }]);
      return; // keep input + attachments so nothing is silently dropped
    }

    const entryId = "cot-" + Math.random().toString(36).substring(2, 11);
    setCoTEntries((prev) => [...prev, {
      id: entryId,
      prompt: text,
      timestamp: Date.now(),
      status: "loading",
    }]);
    setInputText("");
    setIsCoTLoading(true);

    let timeoutId: any;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error("cot_timeout")), COT_TIMEOUT_MS);
    });

    try {
      const base = fastApiBaseUrl.trim().replace(/\/+$/, "");
      if (!base) throw new Error("No Base URL configured — set it in Settings first.");

      const res: Response = await Promise.race([
        fetch(`${base}/chat/cot`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": fastApiKey,
            "X-CoT-Activate": "diagnostic-replay",
          },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        }),
        timeoutPromise,
      ]);

      if (res.status === 403) throw new Error("403 — this endpoint requires the primary API key (guest keys are rejected server-side).");
      if (res.status === 401) throw new Error("401 — API key rejected as invalid by the server.");
      if (!res.ok) throw new Error(`Server returned HTTP ${res.status} for /chat/cot.`);

      const data: any = await res.json();

      if (data && data.error === "cot_parse_failed") {
        updateCoTEntry(entryId, {
          status: "parse_failed",
          errorDetail: data.detail || "The server could not parse the model's output into reasoning/reply.",
          rawOutput: typeof data.raw_output === "string" ? data.raw_output : JSON.stringify(data.raw_output ?? data, null, 2),
        });
      } else if (data && data.reply && typeof data.reply.content === "string") {
        updateCoTEntry(entryId, {
          status: "ok",
          reasoning: {
            content: data.reasoning?.content ?? "(no reasoning content returned)",
            note: data.reasoning?.note,
          },
          reply: data.reply.content,
          measuredTokens: data.measured_tokens,
          durationS: data.duration_s,
        });
        addLog(`CoT reply received (mode: ${data.mode ?? "?"}, ${data.duration_s ?? "?"}s, ${data.measured_tokens ?? "?"} tokens)`, "api");
      } else {
        updateCoTEntry(entryId, {
          status: "parse_failed",
          errorDetail: "Unexpected /chat/cot response shape — no reply.content present.",
          rawOutput: JSON.stringify(data, null, 2),
        });
      }
    } catch (err: any) {
      if (err?.message === "cot_timeout") {
        updateCoTEntry(entryId, {
          status: "error",
          errorDetail: "CoT request timed out client-side after 950s (the backend allows ~900s of generation). The request may still complete server-side — check the backend log.",
        });
      } else {
        updateCoTEntry(entryId, {
          status: "error",
          errorDetail: `CoT request failed: ${err?.message || "network error"}`,
        });
      }
    } finally {
      clearTimeout(timeoutId);
      setIsCoTLoading(false);
    }
  };

  const filteredMessages = messages.filter((m) => {
    if (!searchQuery.trim()) return true;
    return m.text.toLowerCase().includes(searchQuery.toLowerCase());
  });

  // Chronological transcript: normal messages and CoT exchanges interleaved
  // by time. Previously CoT entries rendered as one block after ALL messages,
  // which buried the bot's latest real replies under older CoT prompts.
  const transcriptItems = useMemo(() => {
    const items: Array<{ kind: "msg"; msg: ChatMessage } | { kind: "cot"; entry: CoTEntry }> = [
      ...filteredMessages.map((msg) => ({ kind: "msg" as const, msg })),
      ...cotEntries.map((entry) => ({ kind: "cot" as const, entry })),
    ];
    return items.sort((a, b) => {
      const ta = a.kind === "msg" ? a.msg.timestamp : a.entry.timestamp;
      const tb = b.kind === "msg" ? b.msg.timestamp : b.entry.timestamp;
      return ta - tb;
    });
  }, [filteredMessages, cotEntries]);

  const formatMessageTime = (time: number) => {
    return new Date(time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const isSameDay = (time1: number, time2: number) => {
    const d1 = new Date(time1);
    const d2 = new Date(time2);
    return (
      d1.getFullYear() === d2.getFullYear() &&
      d1.getMonth() === d2.getMonth() &&
      d1.getDate() === d2.getDate()
    );
  };

  const formatHeaderDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);

    if (
      date.getFullYear() === today.getFullYear() &&
      date.getMonth() === today.getMonth() &&
      date.getDate() === today.getDate()
    ) {
      return "Today";
    }

    if (
      date.getFullYear() === yesterday.getFullYear() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getDate() === yesterday.getDate()
    ) {
      return "Yesterday";
    }

    return date.toLocaleDateString([], {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const renderMessage = (msg: ChatMessage) => {
    const isMe = msg.sender === "user";
    const isSelected = selectedMessageId === msg.id;

    return (
      <div className={`flex flex-col relative group w-full ${isMe ? "items-end" : "items-start"}`}>
        {isSelected && (
          <div className={`absolute top-1/2 -translate-y-1/2 flex items-center space-x-1 z-20 ${isMe ? "right-[77%] mr-2" : "left-[77%] ml-2"}`}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setReplyingTo(msg);
                setSelectedMessageId(null);
              }}
              title="Reply"
              className="p-1.5 bg-white border border-slate-200 text-slate-500 rounded-full shadow-md hover:bg-slate-50 cursor-pointer"
            >
              <CornerUpLeft className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                copyToClipboard(msg.text);
                setSelectedMessageId(null);
              }}
              title="Copy Message"
              className="p-1.5 bg-white border border-slate-200 text-slate-500 rounded-full shadow-md hover:bg-slate-50 cursor-pointer"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
        <div
          onClick={(e) => {
            e.stopPropagation();
            setSelectedMessageId(isSelected ? null : msg.id);
          }}
          className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-3.5 py-2 shadow-sm text-[14.5px] relative cursor-pointer transition-all ${
            isSelected ? "ring-2 ring-emerald-500 ring-offset-1" : ""
          } ${isMe ? "bg-[#d9fdd3] text-[#111b21] rounded-tr-none" : "bg-white text-[#111b21] rounded-tl-none"}`}
        >
          <div className={`absolute top-0 w-2 h-2 ${isMe ? "right-[-4px] border-t-[8px] border-t-[#d9fdd3] border-r-[4px] border-r-transparent" : "left-[-4px] border-t-[8px] border-t-white border-l-[4px] border-l-transparent"}`} />
          {msg.replyTo && (<div className="bg-black/5 border-l-4 border-emerald-600 rounded-lg p-2 mb-1.5 text-xs text-slate-600"><div className="font-bold text-emerald-700 text-[10px] uppercase">{msg.replyTo.sender === "user" ? "You" : botConfig.name}</div><div className="truncate">{msg.replyTo.text}</div></div>)}
          {msg.attachments?.map(att => (
            <div key={att.id} className="rounded-lg overflow-hidden border border-black/5 mb-1.5 cursor-pointer" onClick={(e) => { e.stopPropagation(); setViewingDocument(att); }}>
              {att.type === "image" ? <img src={att.url || att.base64} className="max-h-64 w-full object-cover" /> : <div className="flex items-center space-x-2 p-2 bg-black/5"><FileText className="w-5 h-5 text-emerald-600" /><span className="text-xs font-bold truncate">{att.name}</span></div>}
            </div>
          ))}
          <div className="break-words leading-relaxed whitespace-pre-wrap select-text">{msg.text}</div>
          <div className="flex items-center justify-end space-x-1 mt-0.5 text-[10.5px] text-slate-400 select-none">
            <span>{formatMessageTime(msg.timestamp)}</span>
            {isMe && (
              <span className={msg.status === "read" ? "text-sky-500" : msg.status === "delivered" ? "text-slate-400" : "text-amber-500"}>
                <CheckCheck className="w-4 h-4" />
              </span>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderCoTEntry = (entry: CoTEntry) => (
    <>
      {/* CoT prompt — persisted user-style bubble, badged so the mode is unmistakable */}
      <div className="flex flex-col w-full items-end">
        <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tr-none px-3.5 py-2 shadow-sm text-[14.5px] bg-[#d9fdd3] text-[#111b21]">
          <div className="flex items-center justify-end mb-0.5"><span className="text-[9px] font-bold uppercase tracking-wide bg-amber-100 text-amber-700 border border-amber-300 rounded px-1.5 py-0.5">CoT</span></div>
          <div className="break-words leading-relaxed whitespace-pre-wrap select-text">{entry.prompt}</div>
          <div className="flex items-center justify-end mt-0.5 text-[10.5px] text-slate-400 select-none"><span>{formatMessageTime(entry.timestamp)}</span></div>
        </div>
      </div>

      {entry.status === "loading" && (
        <div className="flex flex-col w-full items-start">
          <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tl-none border-l-4 border-amber-400 bg-amber-50 px-3.5 py-2.5 shadow-sm">
            <div className="text-[11px] font-semibold text-amber-700 flex items-center space-x-2">
              <Activity className="w-3.5 h-3.5 animate-spin shrink-0" />
              <span>CoT generating — may take several minutes…</span>
            </div>
          </div>
        </div>
      )}

      {entry.status === "ok" && (
        <>
          <div className="flex flex-col w-full items-start">
            <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tl-none border-l-4 border-amber-500 bg-[#2b2410] px-3.5 py-2 shadow-sm">
              <div className="text-[10px] font-bold uppercase tracking-wide text-amber-300 mb-1">🧠 reasoning — self-report, not verified</div>
              {entry.reasoning?.note && (<div className="text-[10px] italic text-amber-200/70 mb-1.5">{entry.reasoning.note}</div>)}
              <div className="break-words leading-relaxed whitespace-pre-wrap select-text text-[13px] text-amber-100">{entry.reasoning?.content}</div>
              <div className="mt-1 text-right text-[10px] text-amber-200/60 select-none">{entry.durationS != null && <span>{entry.durationS}s · </span>}{entry.measuredTokens != null && <span>{entry.measuredTokens} tokens</span>}</div>
            </div>
          </div>
          <div className="flex flex-col w-full items-start">
            <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tl-none px-3.5 py-2 shadow-sm text-[14.5px] bg-white text-[#111b21]">
              <div className="break-words leading-relaxed whitespace-pre-wrap select-text">{entry.reply}</div>
            </div>
          </div>
        </>
      )}

      {entry.status === "parse_failed" && (
        <div className="flex flex-col w-full items-start">
          <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tl-none border-l-4 border-rose-500 bg-rose-50 px-3.5 py-2 shadow-sm">
            <div className="text-[10px] font-bold uppercase tracking-wide text-rose-700 mb-1">⚠️ CoT parse failed</div>
            <div className="text-xs text-rose-600 break-words">{entry.errorDetail}</div>
            {entry.rawOutput && (
              <details className="mt-1.5">
                <summary className="text-[10px] font-bold text-rose-600 cursor-pointer select-none">Inspect raw model output</summary>
                <pre className="mt-1 max-h-40 overflow-y-auto text-[10px] whitespace-pre-wrap font-mono bg-white/80 rounded p-2 text-slate-700 select-text">{entry.rawOutput}</pre>
              </details>
            )}
          </div>
        </div>
      )}

      {entry.status === "error" && (
        <div className="flex flex-col w-full items-start">
          <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tl-none border-l-4 border-rose-500 bg-rose-50 px-3.5 py-2 shadow-sm">
            <div className="text-[10px] font-bold uppercase tracking-wide text-rose-700 mb-1">⚠️ CoT request error</div>
            <div className="text-xs text-rose-600 break-words">{entry.errorDetail}</div>
          </div>
        </div>
      )}
    </>
  );

  return (
    <div className="min-h-[100dvh] w-full flex flex-col font-sans antialiased relative items-center">
      {/* 1. FIXED BACKGROUND LAYER */}
      <div
        className="fixed inset-0 z-0"
        style={{
          backgroundColor: wallpaper.type === "solid" || wallpaper.type === "preset" ? wallpaper.value : "#efeae2",
          backgroundImage: wallpaper.type === "custom" ? `url(${wallpaper.value})` : undefined,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundAttachment: "fixed"
        }}
      />

      {showPatterns && (wallpaper.type === "solid" || wallpaper.type === "preset") && (
        <div className="fixed inset-0 opacity-12 bg-[radial-gradient(#000000_10%,transparent_11%)] bg-[length:14px_14px] pointer-events-none z-0" />
      )}

      {/* Top Green Accent Banner */}
      <div className="h-[127px] w-full bg-[#00a884] fixed top-0 left-0 z-0 hidden md:block" />

      {/* Main Container */}
      <div className="flex-1 w-full max-w-[1600px] md:px-6 md:py-5 relative z-10 flex flex-col md:flex-row justify-center">

        {/* Actual App Window */}
        <div className="w-full flex flex-col md:flex-row bg-transparent md:bg-[#eae6df] rounded-none md:rounded-2xl shadow-2xl md:border border-gray-300/40 min-h-[100dvh] md:min-h-0 overflow-visible relative">

          {/* 1. LEFT PANEL: SIDEBAR */}
          <div className={`w-full md:w-[410px] shrink-0 bg-white flex flex-col border-r border-slate-200/90 transition-all duration-300 z-50 ${isMobileSidebarOpen ? "flex h-[100dvh] sticky top-0" : "hidden md:flex h-full"
            }`}>
            {showSettings ? (
              <>
                <div className="bg-[#008069] text-white px-4 py-[20px] flex items-center space-x-6 shrink-0 shadow-sm">
                  <button onClick={() => setShowSettings(false)} className="p-1 hover:bg-emerald-800 rounded-full transition cursor-pointer">
                    <ArrowLeft className="w-5 h-5 text-white" />
                  </button>
                  <h2 className="text-base font-semibold tracking-wide">Settings</h2>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/40">
                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-3.5 shadow-xs">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                      <h4 className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider">
                        <Globe className="w-4 h-4 mr-1.5 text-[#00a884]" /> Laptop WiFi Connection
                      </h4>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-tight ${isConnected ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"}`}>
                        {isConnected ? "Connected" : "Disconnected"}
                      </span>
                    </div>

                    <div className="space-y-3.5 text-xs font-sans">
                      <div>
                        <label className="block text-[10px] text-slate-500 font-bold mb-1 uppercase tracking-tight">Laptop Base URL</label>
                        <input type="text" value={fastApiBaseUrl} onChange={(e) => setFastApiBaseUrl(e.target.value)} className="w-full bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg focus:outline-none focus:border-[#00a884] text-xs font-mono text-slate-800" placeholder="e.g. http://192.168.1.5:8000/" />
                      </div>
                      <div>
                        <label className="block text-[10px] text-slate-500 font-bold mb-1 uppercase tracking-tight">API Key (X-API-Key)</label>
                        <input type="password" value={fastApiKey} onChange={(e) => setFastApiKey(e.target.value)} className="w-full bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg focus:outline-none focus:border-[#00a884] text-xs font-mono text-slate-800" placeholder="Enter X-API-Key from secret_key.py" />
                      </div>
                      <button type="button" onClick={testLocalConnection} disabled={isTestingConnection} className="w-full bg-[#128c7e] hover:bg-[#075e54] text-white py-2 rounded-lg font-bold text-[10px] transition shadow-xs flex items-center justify-center space-x-1.5 cursor-pointer">
                        <Activity className={`w-3.5 h-3.5 ${isTestingConnection ? "animate-spin" : ""}`} />
                        <span>{isTestingConnection ? "Pinging Laptop..." : "Test Connection"}</span>
                      </button>
                    </div>
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-2.5 shadow-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider"><Brain className="w-4 h-4 mr-1.5 text-amber-500" /> CoT Diagnostic Mode (primary key only)</span>
                      <button type="button" onClick={() => setCotMode(!cotMode)} className={`relative inline-flex h-5 w-10 shrink-0 items-center rounded-full transition-colors cursor-pointer ${cotMode ? "bg-amber-500" : "bg-slate-300"}`}>
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${cotMode ? "translate-x-5" : "translate-x-1"}`} />
                      </button>
                    </div>
                    <p className="text-[10px] leading-relaxed text-slate-500">Sends messages to a special diagnostic endpoint that also returns the model's self-narrated reasoning before the reply. Takes several minutes per message. Requires the primary API key (guest keys are rejected server-side). The reasoning is a self-report, NOT verified to reflect the model's actual process.</p>
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-3 shadow-xs">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                      <h4 className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider">
                        <Bell className="w-4 h-4 mr-1.5 text-[#00a884]" /> Notifications
                      </h4>
                      <span className="text-[10px] text-slate-400 font-medium">Alerts & Badges</span>
                    </div>
                    <p className="text-[10.5px] leading-relaxed text-slate-500">
                      Receive notifications when {botConfig.name} replies while this tab or app is in the background.
                    </p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={async () => {
                          const ok = await requestNotificationPermissions();
                          if (ok) {
                            showToast("Notifications enabled!");
                          } else {
                            showToast("Permission status checked");
                          }
                        }}
                        className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-700 py-1.5 px-3 rounded-lg font-bold text-[10px] transition cursor-pointer"
                      >
                        Enable Notifications
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          await triggerNotification(botConfig.name, "Test notification: BotBot alerts are working!");
                          showToast("Test notification dispatched!");
                        }}
                        className="flex-1 bg-[#128c7e] hover:bg-[#075e54] text-white py-1.5 px-3 rounded-lg font-bold text-[10px] transition shadow-xs flex items-center justify-center space-x-1 cursor-pointer"
                      >
                        <Bell className="w-3 h-3" />
                        <span>Test Notification</span>
                      </button>
                    </div>
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-3.5 shadow-xs">
                    <div className="flex justify-between items-center"><h4 className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider"><Sparkles className="w-4 h-4 mr-1.5 text-[#00a884]" /> Customize Bot Identity</h4></div>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-3.5">
                        <div className="w-14 h-14 rounded-full bg-white shadow-inner border border-slate-200 flex items-center justify-center shrink-0 overflow-hidden">{renderAvatar(botConfig.avatar, "w-14 h-14")}</div>
                        <div className="flex-1 space-y-1.5">
                          <input type="file" accept="image/*" id="bot-profile-pic-uploader" className="hidden" onChange={handleAvatarUpload} />
                          <label htmlFor="bot-profile-pic-uploader" className="inline-flex items-center space-x-1 px-3 py-1.5 bg-[#128c7e] hover:bg-[#075e54] text-white text-[10px] font-bold rounded-lg cursor-pointer transition shadow-xs"><Upload className="w-3.5 h-3.5" /><span>Upload Photo</span></label>
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3.5 pt-1.5">
                      <input type="text" value={botConfig.name} onChange={(e) => setBotConfig((prev) => ({ ...prev, name: e.target.value }))} className="w-full bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg text-xs" />
                      <input type="text" value={botConfig.nickname} onChange={(e) => setBotConfig((prev) => ({ ...prev, nickname: e.target.value }))} className="w-full bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg text-xs" />
                    </div>
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-2.5 shadow-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider"><Palette className="w-4 h-4 mr-1.5 text-emerald-500" /> Appearance & Themes</span>
                      <button onClick={() => setShowWallpaperDrawer(!showWallpaperDrawer)} className="px-3 py-1 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded-lg text-[10px] font-bold border border-slate-200">{showWallpaperDrawer ? "Close" : "Open Settings"}</button>
                    </div>
                    {showWallpaperDrawer && <div className="p-1 pt-3 border-t border-slate-100 animate-in slide-in-from-top-4 duration-200"><WallpaperSettings currentWallpaper={wallpaper} onWallpaperChange={(w) => setWallpaper(w)} showPatterns={showPatterns} onTogglePatterns={() => setShowPatterns(!showPatterns)} /></div>}
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 space-y-3 shadow-xs">
                    <span className="text-xs font-bold text-slate-800 flex items-center uppercase tracking-wider"><Database className="w-4 h-4 mr-1.5 text-blue-500" /> Session Switcher</span>
                    {previousSessions.length > 1 && (
                      <div className="flex flex-wrap gap-1.5">{previousSessions.map(id => (<button key={id} onClick={() => switchSession(id)} className={`px-2 py-1 rounded text-[10px] font-mono border ${id === sessionId ? "bg-blue-50 border-blue-200 text-blue-600 font-bold" : "bg-slate-50 border-slate-100 text-slate-500"}`}>{id.replace("session-", "")}</button>))}</div>
                    )}
                    <div className="flex space-x-2 mt-2">
                      <input ref={sessionInputRef} type="text" defaultValue={sessionId} onKeyDown={(e) => { if (e.key === "Enter") switchSession((e.target as HTMLInputElement).value); }} className="flex-1 bg-slate-50 border border-slate-200 px-2.5 py-1.5 rounded-lg text-xs font-mono" placeholder="Session name..." />
                      <button onClick={() => switchSession(sessionInputRef.current?.value ?? "")} className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-[10px] font-bold uppercase">Switch</button>
                    </div>
                  </div>

                  <div className="bg-white p-4 rounded-xl border border-slate-200/60 flex items-center justify-between shadow-xs">
                    <button onClick={handleNewChat} className="w-full px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-600 border border-emerald-200 rounded-lg text-xs font-bold transition">New Chat</button>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="bg-[#f0f2f5] px-4 py-3 flex items-center justify-between border-b border-slate-200 shrink-0">
                  <div className="flex items-center space-x-2.5">
                    <div className="w-10 h-10 rounded-full bg-[#128c7e] text-white flex items-center justify-center font-bold text-lg shadow-sm shrink-0">{userName[0].toUpperCase()}</div>
                    <h2 onClick={() => setIsEditingName(true)} className="text-xs font-bold text-slate-800 cursor-pointer">{userName}</h2>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button onClick={() => setShowSearchBar((s) => !s)} className={`p-1.5 rounded-full transition cursor-pointer ${showSearchBar ? "bg-emerald-100 text-emerald-700" : "hover:bg-slate-200/70 text-slate-700"}`}><Search className="w-5 h-5" /></button>
                    <button onClick={() => setShowSettings(true)} className="p-1.5 rounded-full hover:bg-slate-200/70 transition cursor-pointer text-slate-700"><Settings className="w-5 h-5" /></button>
                  </div>
                </div>
                {showSearchBar && (
                  <div className="px-3 py-2 border-b border-slate-200 bg-white">
                    <div className="flex items-center bg-[#f0f2f5] rounded-lg px-3 py-1.5">
                      <Search className="w-4 h-4 text-slate-400 shrink-0" />
                      <input
                        autoFocus
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search messages..."
                        className="w-full bg-transparent px-2 text-xs focus:outline-none"
                      />
                      {searchQuery && (
                        <button onClick={() => setSearchQuery("")} className="text-slate-400 shrink-0"><X className="w-4 h-4" /></button>
                      )}
                    </div>
                  </div>
                )}
                <div className="flex-1 overflow-y-auto bg-white select-none">
                  <div className="text-[11px] text-slate-400 px-4 pt-3 pb-1 uppercase font-bold tracking-wider">Conversations</div>
                  <div onClick={() => setIsMobileSidebarOpen(false)} className="flex items-center space-x-3.5 px-4 py-3.5 hover:bg-[#f0f2f5] transition bg-emerald-50/10 border-b border-slate-150/60 cursor-pointer">
                    {renderAvatar(botConfig.avatar, "w-12 h-12")}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between"><h4 className="font-bold text-sm text-slate-800 truncate">{botConfig.name}</h4><span className="text-[10px] text-slate-400 font-medium">{messages.length > 0 ? formatMessageTime(messages[messages.length - 1].timestamp) : "Now"}</span></div>
                      <p className="text-[11.5px] text-slate-500 truncate flex items-center">{messages.length > 0 && messages[messages.length - 1].sender === "user" && <CheckCheck className="w-3.5 h-3.5 mr-1 text-sky-500" />}{messages.length > 0 ? messages[messages.length - 1].text : "No messages yet"}</p>
                    </div>
                  </div>
                </div>
              </>
            )}
            <div className="p-3 bg-[#f0f2f5] border-t border-slate-200 text-center text-[10px] text-slate-400 font-mono flex items-center justify-center space-x-1 shrink-0"><Globe className="w-3 h-3 text-[#128c7e]" /><span>Local network connection.</span></div>
          </div>

          {/* 2. MAIN CHAT AREA */}
          <div className={`flex-1 flex flex-col relative min-w-0 ${isMobileSidebarOpen ? "hidden md:flex" : "flex w-full"}`}>

            {/* HEADER */}
            <div className="sticky top-0 bg-[#f0f2f5]/95 backdrop-blur-md px-3 py-2 flex items-center justify-between border-b border-slate-200 shadow-sm shrink-0 z-50 w-full min-h-[60px]" style={{ paddingTop: 'max(0.5rem, env(safe-area-inset-top))' }}>
              <div className="flex items-center space-x-2.5 min-w-0 flex-1 mr-2">
                <button onClick={() => setIsMobileSidebarOpen(true)} className="p-1.5 hover:bg-slate-200 rounded-md transition shrink-0"><Menu className="w-6 h-6 text-slate-600" /></button>
                <div className="shrink-0">{renderAvatar(botConfig.avatar, "w-10 h-10")}</div>
                <div className="min-w-0 leading-tight">
                  <h3 className="font-bold text-[16px] text-slate-800 truncate leading-none mb-0.5">{botConfig.name}</h3>
                  <p className={`text-[10.5px] flex items-center font-medium truncate ${isConnected ? "text-[#00a884]" : "text-rose-500"}`}>
                    <span className={`w-1.5 h-1.5 rounded-full mr-2 shrink-0 ${isConnected ? "bg-emerald-500" : "bg-rose-500"}`}></span>
                    <span className="truncate max-w-[130px] sm:max-w-[240px]">{isConnected ? (fastApiBaseUrl.includes("trycloudflare") ? "Tunnel Connected" : `LAN: ${fastApiBaseUrl}`) : "Offline (Queueing locally)"}</span>
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-1 shrink-0">
                <button onClick={captureLongScreenshot} title="Capture Full Chat Screenshot" className="p-2 rounded-full text-slate-600 hover:bg-slate-200 transition cursor-pointer">
                  <Camera className="w-5 h-5" />
                </button>
                <button onClick={() => setShowChatWallpaperDrawer(!showChatWallpaperDrawer)} title="Wallpaper Settings" className={`p-2 rounded-full transition cursor-pointer ${showChatWallpaperDrawer ? "bg-emerald-100 text-[#128c7e]" : "text-[#128c7e] hover:bg-slate-200"}`}>
                  <Palette className="w-5 h-5" />
                </button>
              </div>
            </div>

            {cotMode && (
              <div className="bg-amber-100 border-b border-amber-300 text-amber-800 text-[11px] font-semibold px-3 py-1.5 flex items-center justify-center space-x-2">
                <Brain className="w-3.5 h-3.5 shrink-0" />
                <span className="text-center">CoT diagnostic mode — reasoning is a self-report; replies may take several minutes</span>
              </div>
            )}

            {showChatWallpaperDrawer && (
              <div className="fixed top-[60px] left-0 right-0 md:absolute md:left-0 md:right-0 bg-[#f0f2f5] border-b border-slate-200 p-4 z-30 shadow-md max-h-[360px] overflow-y-auto">
                <div className="flex justify-between items-center mb-3"><h4 className="text-xs font-bold text-slate-700 uppercase flex items-center"><Palette className="w-4 h-4 mr-1.5 text-emerald-600" /> Wallpaper Settings</h4><button onClick={() => setShowChatWallpaperDrawer(false)} className="text-xs text-rose-500 font-bold px-2 py-1">✕ Close</button></div>
                <WallpaperSettings currentWallpaper={wallpaper} onWallpaperChange={(w) => setWallpaper(w)} showPatterns={showPatterns} onTogglePatterns={() => setShowPatterns(!showPatterns)} />
              </div>
            )}

            {/* FLOW AREA */}
            <div id="chat-flow-container" ref={chatFlowRef} onClick={() => setSelectedMessageId(null)} className="flex-1 flex flex-col space-y-4 px-3 py-4 z-10">
              <div className="bg-[#ffe596]/70 backdrop-blur-xs text-[#6b5300] p-2 px-3.5 rounded-lg text-[11px] shadow-xs flex items-center justify-center space-x-2 border border-[#ffe596]/40 max-w-xs mx-auto mb-2"><Globe className="w-3.5 h-3.5 shrink-0" /><span className="font-semibold text-center select-none">Private — stays on your local network</span></div>
              {transcriptItems.map((item, idx) => {
                const itemTs = item.kind === "msg" ? item.msg.timestamp : item.entry.timestamp;
                const prevItem = idx > 0 ? transcriptItems[idx - 1] : null;
                const prevTs = prevItem ? (prevItem.kind === "msg" ? prevItem.msg.timestamp : prevItem.entry.timestamp) : null;
                const showDateDivider = prevTs === null || !isSameDay(prevTs, itemTs);
                return (
                  <React.Fragment key={item.kind === "msg" ? item.msg.id : item.entry.id}>
                    {showDateDivider && (
                      <div className="flex justify-center my-3 select-none date-separator-divider">
                        <span className="bg-[#e1f3fc]/90 backdrop-blur-xs text-[#025179] text-[11px] font-semibold px-3 py-1 rounded-lg uppercase tracking-wider shadow-xs border border-sky-200/50">
                          {formatHeaderDate(itemTs)}
                        </span>
                      </div>
                    )}
                    {item.kind === "msg" ? renderMessage(item.msg) : renderCoTEntry(item.entry)}
                  </React.Fragment>
                );
              })}
              {isTyping && (
                <div className="flex items-start">
                  <div className="bg-white rounded-2xl rounded-tl-none px-4 py-3 shadow-sm flex items-center space-x-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} className="h-4" />
            </div>

            {/* INPUT BAR */}
            <div className="sticky bottom-0 z-40 w-full bg-[#f0f2f5] border-t border-slate-200 flex flex-col pb-4 md:pb-3">
              {replyingTo && (
                <div className="mx-3 mt-2 mb-1 bg-white p-2 px-3 rounded-lg border-l-4 border-emerald-600 flex justify-between shadow-sm">
                  <div className="min-w-0 flex-1">
                    <span className="block text-[10px] font-bold text-emerald-600 uppercase">Replying to {replyingTo.sender === "user" ? "You" : botConfig.name}</span>
                    <p className="text-xs text-slate-600 truncate">{replyingTo.text}</p>
                  </div>
                  <button onClick={() => setReplyingTo(null)} className="p-1 text-slate-400"><X className="w-4 h-4" /></button>
                </div>
              )}

              {attachedFiles.length > 0 && (
                <div className="mx-3 mt-2 flex gap-2 overflow-x-auto p-2 bg-white rounded-lg border border-slate-200 shadow-sm">
                  {attachedFiles.map(f => (
                    <div key={f.id} className="relative shrink-0">
                      <div className="w-12 h-12 rounded border bg-slate-50 flex items-center justify-center overflow-hidden">
                        {f.type === "image" ? <img src={f.url} className="object-cover w-full h-full" /> : <FileText className="w-6 h-6 text-emerald-600" />}
                      </div>
                      <button onClick={() => setAttachedFiles(prev => prev.filter(af => af.id !== f.id))} className="absolute -top-1 -right-1 bg-rose-500 text-white rounded-full p-0.5 shadow-sm"><X className="w-3 h-3" /></button>
                    </div>
                  ))}
                </div>
              )}

              <form onSubmit={cotMode ? handleSendCoT : handleSendMessage} className="px-3 py-2 flex items-center space-x-2">
                <button type="button" onClick={() => fileInputRef.current?.click()} className="p-1.5 hover:bg-slate-200 rounded-full transition text-[#128c7e] shrink-0">
                  <Plus className="w-6 h-6" />
                </button>
                <input type="file" ref={fileInputRef} className="hidden" accept="image/*,.pdf,.docx,.txt" multiple onChange={handleFileChange} />

                <div className="flex-1 bg-white rounded-xl px-4 py-2 shadow-sm border border-slate-200 flex items-center">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Type a message..."
                    className="w-full text-sm text-[#111b21] bg-transparent focus:outline-none placeholder:text-slate-400"
                  />
                </div>

                <button
                  type="submit"
                  disabled={(!inputText.trim() && attachedFiles.length === 0) || (cotMode && isCoTLoading)}
                  className={`p-2.5 rounded-xl flex items-center justify-center text-white shrink-0 transition-all ${(inputText.trim() || attachedFiles.length > 0) ? "bg-[#00a884] shadow-md active:scale-95" : "bg-gray-300 opacity-60"}`}
                >
                  <Send className="w-5 h-5" />
                </button>
              </form>
            </div>
          </div>

          {showInspectorPanel && (
            <div className="w-[360px] border-l border-slate-200 bg-slate-950 shrink-0 hidden lg:flex flex-col h-full z-20">
              <AIEngineInspector logs={logs} payloadData={activePayload} localConfig={{ endpoint: fastApiBaseUrl, model: "FastAPI" }} testLocalConnection={testLocalConnection} />
            </div>
          )}
        </div>
      </div>

      {/* ATTACHMENT / DOCUMENT VIEWER MODAL */}
      {viewingDocument && (
        <div
          className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center p-4"
          onClick={() => setViewingDocument(null)}
        >
          <div
            className="bg-white rounded-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-3 border-b border-slate-200 sticky top-0 bg-white">
              <span className="text-sm font-bold text-slate-800 truncate pr-2">{viewingDocument.name}</span>
              <button onClick={() => setViewingDocument(null)} className="p-1.5 rounded-full hover:bg-slate-100 shrink-0">
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            <div className="p-4">
              {viewingDocument.type === "image" ? (
                <img src={viewingDocument.url || viewingDocument.base64} className="w-full h-auto rounded-lg" />
              ) : (
                <pre className="text-xs whitespace-pre-wrap text-slate-700 font-mono">
                  {viewingDocument.textContents || "No preview available for this file."}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
      {/* FLOATING TOAST NOTIFICATION */}
      {toastMessage && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-[110] bg-slate-900/95 text-white text-xs font-medium px-4 py-2.5 rounded-full shadow-2xl backdrop-blur-md flex items-center space-x-2 animate-in fade-in slide-in-from-bottom-2 border border-slate-700/50">
          <Check className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{toastMessage}</span>
        </div>
      )}
    </div>
  );
}