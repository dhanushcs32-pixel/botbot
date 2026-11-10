import express from "express";
import http from "http";
import { WebSocketServer, WebSocket } from "ws";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface EncryptedMessage {
  id: string;
  sender: "alice" | "bob";
  receiver: "alice" | "bob";
  ciphertext: string;
  iv: string;
  timestamp: number;
}

interface UserSession {
  username: "alice" | "bob";
  publicKey: any; // JWK representation of ECDH public key
  ws: WebSocket;
}

async function startServer() {
  const app = express();
  const server = http.createServer(app);
  const PORT = 3000;

  // Body parsers with high limit for images/files
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));

  // Logging middleware
  app.use((req, res, next) => {
    console.log(`[HTTP] ${req.method} ${req.url}`);
    next();
  });

  // Setup Gemini SDK
  const geminiKey = process.env.GEMINI_API_KEY;
  let ai: any = null;
  if (geminiKey) {
    try {
      ai = new GoogleGenAI({
        apiKey: geminiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });
      console.log("Gemini SDK initialized successfully.");
    } catch (err) {
      console.error("Failed to initialize Gemini SDK:", err);
    }
  } else {
    console.warn("GEMINI_API_KEY is not defined in environment variables. Gemini features will require setup.");
  }

  // Gemini AI Chat Route (Original)
  app.post("/api/chat/gemini", async (req, res) => {
    const { message, image, pdfText, systemInstruction, history } = req.body;

    if (!ai) {
      return res.status(503).json({
        error: "Gemini AI is not configured. Please add GEMINI_API_KEY in the Settings > Secrets panel on Google AI Studio."
      });
    }

    try {
      // Build contents array starting with history if any, then add current parts
      let contents: any[] = [];
      
      // Add chat history if present
      if (history && Array.isArray(history)) {
        contents = history.map((turn: any) => {
          return {
            role: turn.sender === "user" ? "user" : "model",
            parts: [{ text: turn.text }]
          };
        });
      }

      // Prepare current message parts
      const parts: any[] = [];
      
      // If there is PDF/Document text context, inject it
      if (pdfText) {
        parts.push({
          text: `[DOCUMENT CONTENT ATTACHED BY USER]\n${pdfText}\n\nPlease analyze this document and use it as reference context to answer the user's inquiry.`
        });
      }

      // If there is an image, construct the inlineData object
      if (image && image.base64 && image.mimeType) {
        let base64Data = image.base64;
        if (base64Data.includes(";base64,")) {
          base64Data = base64Data.split(";base64,")[1];
        }
        parts.push({
          inlineData: {
            data: base64Data,
            mimeType: image.mimeType
          }
        });
      }

      // Add the user message text
      parts.push({ text: message || "Analyze the attachments." });

      // Append current turn
      contents.push({
        role: "user",
        parts: parts
      });

      console.log(`Generating content using gemini-1.5-flash with ${contents.length} turns in history...`);
      const response = await ai.models.generateContent({
        model: "gemini-1.5-flash",
        contents: contents,
        config: {
          systemInstruction: systemInstruction || "You are a helpful WhatsApp AI Bot. You can discuss any topic, summarize text, describe attached images, and guide the user.",
          temperature: 0.7,
        }
      });

      res.json({
        text: response.text || "I processed your request but didn't generate any text response.",
        rawResponse: response
      });

    } catch (err: any) {
      console.error("Gemini Generation Error:", err);
      res.status(500).json({
        error: err.message || "An unexpected error occurred during Gemini AI processing."
      });
    }
  });

  // Compatibility routes for the React frontend (App.tsx)
  app.get("/", (req, res, next) => {
    // If it's not a dev environment, serve the index.html from dist
    // In dev environment, this will be handled by Vite middleware later
    if (process.env.NODE_ENV === "production") {
      return res.sendFile(path.join(process.cwd(), "dist", "index.html"));
    }
    // If it's API health check call from frontend
    if (req.headers.accept?.includes("application/json")) {
      return res.json({ status: "online", message: "Node.js Gemini Server is running" });
    }
    next();
  });

  app.post("/chat", async (req, res) => {
    const { message, session_id, attachments } = req.body;
    console.log(`[Chat] Session: ${session_id}, Message: ${message}`);

    if (!ai) {
      return res.json({
        reply: "⚠️ Gemini AI is not configured on the server. Please set GEMINI_API_KEY in the .env file.",
        mode: "error"
      });
    }

    try {
      const response = await ai.models.generateContent({
        model: "gemini-1.5-flash",
        contents: [{ role: "user", parts: [{ text: message }] }],
        config: {
          systemInstruction: "You are a helpful WhatsApp AI Bot. Answer the user's questions clearly.",
          temperature: 0.7,
        }
      });

      res.json({
        reply: response.text || "I processed your request but didn't generate any text response.",
        mode: "gemini"
      });
    } catch (err: any) {
      console.error("Gemini Generation Error:", err);
      res.json({
        reply: "⚠️ Error generating response: " + (err.message || "Unknown error"),
        mode: "error"
      });
    }
  });

  app.post("/search", (req, res) => {
    const { query } = req.body;
    console.log(`[Search] Query: ${query}`);
    res.json({
      results: [`This is a simulated search result for: "${query}". Real search index not implemented in Node backend.`],
      count: 1
    });
  });

  app.post("/summarize", (req, res) => {
    res.json({ summary: "Summary feature not implemented in Node backend." });
  });

  // In-memory database
  let messages: EncryptedMessage[] = [];
  const activeSessions = new Map<WebSocket, "alice" | "bob">();
  const registeredUsers = new Map<"alice" | "bob", { publicKey: any; online: boolean }>();

  // Ensure default structure for alice and bob exists
  registeredUsers.set("alice", { publicKey: null, online: false });
  registeredUsers.set("bob", { publicKey: null, online: false });

  // API endpoints (if needed for diagnostics, but we rely on WebSockets for real-time sync)
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  app.get("/api/state", (req, res) => {
    res.json({
      registeredUsers: Array.from(registeredUsers.entries()).map(([username, data]) => ({
        username,
        online: data.online,
        hasKey: !!data.publicKey,
      })),
      encryptedMessagesCount: messages.length,
    });
  });

  // Setup WebSocket Server
  const wss = new WebSocketServer({ server });

  function broadcast(payload: any) {
    const dataString = JSON.stringify(payload);
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(dataString);
      }
    });
  }

  function getPresenceData() {
    return {
      type: "presence",
      alice: {
        online: registeredUsers.get("alice")?.online || false,
        publicKey: registeredUsers.get("alice")?.publicKey || null,
      },
      bob: {
        online: registeredUsers.get("bob")?.online || false,
        publicKey: registeredUsers.get("bob")?.publicKey || null,
      },
    };
  }

  wss.on("connection", (ws: WebSocket) => {
    console.log("WebSocket connection established");

    // Send initial history and presence state to the new connection
    ws.send(JSON.stringify({ type: "history", messages }));
    ws.send(JSON.stringify(getPresenceData()));

    ws.on("message", (messageStr: string) => {
      try {
        const payload = JSON.parse(messageStr);
        console.log("Received WebSocket event:", payload.type);

        switch (payload.type) {
          case "register": {
            const { username, publicKey } = payload;
            if (username === "alice" || username === "bob") {
              activeSessions.set(ws, username);
              registeredUsers.set(username, { publicKey, online: true });

              console.log(`User registered: ${username}`);
              
              // Broadcast updated presence to everyone
              broadcast(getPresenceData());
            }
            break;
          }

          case "message": {
            const { sender, receiver, ciphertext, iv, timestamp } = payload.message;
            if ((sender === "alice" || sender === "bob") && (receiver === "alice" || receiver === "bob")) {
              const newMessage: EncryptedMessage = {
                id: Math.random().toString(36).substring(2, 11),
                sender,
                receiver,
                ciphertext,
                iv,
                timestamp: timestamp || Date.now(),
              };

              // Limit message history to last 100 messages to prevent infinite memory growth
              messages.push(newMessage);
              if (messages.length > 100) {
                messages.shift();
              }

              console.log(`Encrypted message routed from ${sender} to ${receiver}`);

              // Broadcast message to all connected clients
              broadcast({
                type: "message",
                message: newMessage,
              });
            }
            break;
          }

          case "clear_history": {
            messages = [];
            console.log("Cleared message history");
            broadcast({ type: "history", messages: [] });
            break;
          }

          case "reset_keys": {
            registeredUsers.set("alice", { publicKey: null, online: false });
            registeredUsers.set("bob", { publicKey: null, online: false });
            activeSessions.clear();
            messages = [];
            console.log("Reset all keys and history");
            broadcast(getPresenceData());
            broadcast({ type: "history", messages: [] });
            break;
          }

          default:
            console.warn("Unknown socket message type:", payload.type);
        }
      } catch (err) {
        console.error("Error processing WebSocket message:", err);
      }
    });

    ws.on("close", () => {
      const username = activeSessions.get(ws);
      if (username) {
        console.log(`User disconnected: ${username}`);
        activeSessions.delete(ws);
        const userState = registeredUsers.get(username);
        if (userState) {
          userState.online = false;
          registeredUsers.set(username, userState);
        }
        broadcast(getPresenceData());
      }
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  server.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start fullstack server:", err);
});
