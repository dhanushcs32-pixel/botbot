/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { 
  Terminal, 
  Cpu, 
  Copy, 
  Check, 
  Network,
  Settings,
  HelpCircle
} from "lucide-react";
import { Clipboard } from '@capacitor/clipboard';

interface AIEngineInspectorProps {
  logs: Array<{ time: string; type: "info" | "api" | "error"; text: string }>;
  payloadData: any;
  localConfig: { endpoint: string; model: string };
  testLocalConnection: () => Promise<boolean>;
}

export default function AIEngineInspector({
  logs,
  payloadData,
  localConfig,
  testLocalConnection
}: AIEngineInspectorProps) {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"payload" | "logs" | "fastapi">("logs");
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "testing" | "success" | "failed">("idle");

  const handleCopyPayload = async () => {
    await Clipboard.write({
      string: JSON.stringify(payloadData, null, 2)
    });
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleTestConnection = async () => {
    setConnectionStatus("testing");
    const ok = await testLocalConnection();
    setConnectionStatus(ok ? "success" : "failed");
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 rounded-2xl border border-slate-800 shadow-xl overflow-hidden font-mono text-xs">
      {/* Inspector Header */}
      <div className="bg-slate-900 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="font-bold uppercase tracking-wider text-emerald-400">WiFi Node Inspector</span>
        </div>
        <div className="flex items-center space-x-1.5 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-[10px] text-slate-400">FastAPI WiFi Companion</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-slate-900/60 border-b border-slate-800 flex shrink-0">
        <button
          onClick={() => setActiveTab("logs")}
          className={`flex-1 py-2 px-3 text-center border-b-2 transition-all cursor-pointer ${
            activeTab === "logs" 
              ? "border-emerald-500 text-white bg-slate-800/40" 
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          🔍 Activity logs
        </button>
        <button
          onClick={() => setActiveTab("payload")}
          className={`flex-1 py-2 px-3 text-center border-b-2 transition-all cursor-pointer ${
            activeTab === "payload" 
              ? "border-emerald-500 text-white bg-slate-800/40" 
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          📨 Request Payloads
        </button>
        <button
          onClick={() => setActiveTab("fastapi")}
          className={`flex-1 py-2 px-3 text-center border-b-2 transition-all cursor-pointer ${
            activeTab === "fastapi" 
              ? "border-emerald-500 text-white bg-slate-800/40" 
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          📡 WiFi Setup Guide
        </button>
      </div>

      {/* Tab Panels */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {activeTab === "logs" && (
          <div className="space-y-2 h-full flex flex-col">
            <div className="flex justify-between items-center text-slate-400 text-[10px] pb-1 border-b border-slate-900 shrink-0">
              <span>REAL-TIME COMPANION TRACES</span>
              <span>LINES: {logs.length}</span>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-1.5 scrollbar-thin max-h-[380px] text-[11px] select-text">
              {logs.length === 0 ? (
                <div className="text-slate-600 italic py-6 text-center">
                  No activity logs recorded yet. Initiate a chat or connection test to trace!
                </div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className="leading-relaxed hover:bg-slate-900/40 px-1 py-0.5 rounded">
                    <span className="text-slate-600 mr-1.5">[{log.time}]</span>
                    <span className={`font-semibold mr-1.5 ${
                      log.type === "error" ? "text-rose-400" :
                      log.type === "api" ? "text-blue-400" : "text-emerald-400"
                    }`}>
                      {log.type.toUpperCase()}:
                    </span>
                    <span className="text-slate-300">{log.text}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === "payload" && (
          <div className="space-y-3">
            <div className="flex justify-between items-center pb-2 border-b border-slate-900">
              <span className="text-slate-400 text-[10px]">CURRENT OUTBOUND COMPOSITE</span>
              <button
                onClick={handleCopyPayload}
                disabled={!payloadData}
                className="flex items-center space-x-1 px-2 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white rounded border border-slate-800 transition disabled:opacity-50"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>
            </div>

            {payloadData ? (
              <div className="space-y-2">
                <div className="text-slate-300 leading-normal text-[11px] bg-slate-900/70 p-3 rounded-lg border border-slate-900 overflow-x-auto max-h-[350px]">
                  <pre className="text-sky-300 font-mono whitespace-pre-wrap word-break-all text-xs">
                    {JSON.stringify(payloadData, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-slate-600 italic py-6 text-center">
                Payload data is empty. Send a query or trigger a summary to view JSON formats.
              </div>
            )}
          </div>
        )}

        {activeTab === "fastapi" && (
          <div className="space-y-4 text-xs leading-relaxed text-slate-300">
            <div className="p-3 bg-emerald-950/40 border border-emerald-900/60 rounded-xl">
              <div className="font-bold text-emerald-300 flex items-center mb-1">
                <Network className="w-4 h-4 mr-1.5" /> LAN Connection Rules
              </div>
              <p className="text-[11px] leading-relaxed">
                For the application to query your laptop's FastAPI server, both devices must be on the **same WiFi network**. The laptop must accept local incoming queries.
              </p>
            </div>

            <div className="space-y-3.5 font-sans">
              <h4 className="font-bold text-emerald-400 font-mono text-xs uppercase tracking-wide">
                1. Binding FastAPI Host (CRITICAL)
              </h4>
              <p className="text-[11px] leading-relaxed text-slate-300">
                Do not bind your server to `127.0.0.1` (localhost) — that limits connections to the laptop itself. Start uvicorn with host `0.0.0.0` so other devices can find it:
              </p>
              <pre className="bg-black/50 p-2.5 rounded font-mono text-[10px] text-white overflow-x-auto">
                uvicorn main:app --host 0.0.0.0 --port 8000 --cors-headers
              </pre>

              <h4 className="font-bold text-sky-400 font-mono text-xs uppercase tracking-wide pt-1">
                2. Exposed Endpoint Specifications
              </h4>
              <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800 space-y-2.5 font-mono text-[10.5px]">
                <div>
                  <span className="text-emerald-400 font-bold">GET /</span>
                  <div className="text-slate-400 text-[9.5px]">Health check. Response:</div>
                  <code className="text-sky-300">{"{\"status\": \"...\"}"}</code>
                </div>
                <div>
                  <span className="text-emerald-400 font-bold">POST /chat</span>
                  <div className="text-slate-400 text-[9.5px]">Chat with history. Body:</div>
                  <code className="text-sky-300">{"{\"message\": \"...\", \"session_id\": \"...\"}"}</code>
                </div>
                <div>
                  <span className="text-emerald-400 font-bold">POST /summarize</span>
                  <div className="text-slate-400 text-[9.5px]">Synthesize chat logs. Body:</div>
                  <code className="text-sky-300">{"{\"session_id\": \"...\", \"force\": true}"}</code>
                </div>
              </div>

              <h4 className="font-bold text-[#128c7e] font-mono text-xs uppercase tracking-wide pt-1">
                3. Android Cleartext Traffic (HTTP)
              </h4>
              <p className="text-[11px] leading-relaxed text-slate-300">
                Since local network addresses use unencrypted HTTP (`http://`), Android blocks them by default. Enabling <strong>Cleartext Traffic</strong> allows these local calls:
              </p>
              <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800 text-[10px] font-mono text-slate-300">
                <div>&lt;application</div>
                <div className="pl-3 text-amber-400">android:usesCleartextTraffic="true"</div>
                <div>... &gt;</div>
              </div>
            </div>

            {/* Local Connection Test */}
            <div className="pt-2">
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="block text-[11px] font-mono text-slate-400">TARGET ENDPOINT</span>
                  <span className="block text-[10px] text-slate-500 truncate max-w-[150px]">{localConfig.endpoint}</span>
                </div>
                <button
                  onClick={handleTestConnection}
                  disabled={connectionStatus === "testing"}
                  className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded transition flex items-center space-x-1.5 shadow-sm"
                >
                  <Settings className={`w-3.5 h-3.5 ${connectionStatus === "testing" ? "animate-spin" : ""}`} />
                  <span>
                    {connectionStatus === "idle" && "Ping Server"}
                    {connectionStatus === "testing" && "Pinging..."}
                    {connectionStatus === "success" && "Connected!"}
                    {connectionStatus === "failed" && "Offline / Error"}
                  </span>
                </button>
              </div>
              {connectionStatus === "failed" && (
                <p className="text-[10px] text-rose-400 mt-1.5 font-mono">
                  ⚠️ Can't reach Bot — check your laptop is on and connected to the same WiFi. Make sure FastAPI CORS middleware is enabled!
                </p>
              )}
              {connectionStatus === "success" && (
                <p className="text-[10px] text-emerald-400 mt-1.5 font-mono">
                  ✅ Successfully pinged GET / check! FastAPI backend is accessible.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Quick Setup Status Bar */}
      <div className="bg-slate-900 border-t border-slate-800 px-4 py-2 flex justify-between items-center text-[10px] text-slate-400 shrink-0">
        <span className="flex items-center">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
          CORS Diagnostic Engine Active
        </span>
        <span>UTC-07:00</span>
      </div>
    </div>
  );
}
