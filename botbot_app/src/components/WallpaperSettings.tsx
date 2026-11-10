/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { 
  Image, 
  Palette, 
  Upload, 
  Link, 
  Check, 
  Trash2,
  Sliders,
  Sparkles
} from "lucide-react";
import { WallpaperConfig } from "../types";

interface WallpaperSettingsProps {
  currentWallpaper: WallpaperConfig;
  onWallpaperChange: (wallpaper: WallpaperConfig) => void;
  showPatterns: boolean;
  onTogglePatterns: () => void;
}

const PRESET_COLORS = [
  { name: "Sand (Default)", value: "#efeae2", class: "bg-[#efeae2] border-amber-200" },
  { name: "Teal Mint", value: "#cdeae3", class: "bg-[#cdeae3] border-emerald-200" },
  { name: "Lavender", value: "#eae3f2", class: "bg-[#eae3f2] border-violet-200" },
  { name: "Peach Blossom", value: "#fbe4db", class: "bg-[#fbe4db] border-orange-200" },
  { name: "Soft White", value: "#f8fafc", class: "bg-[#f8fafc] border-slate-200" },
  { name: "WhatsApp Dark (Official)", value: "#0b141a", class: "bg-[#0b141a] border-slate-800 text-white" },
  { name: "Dark Emerald", value: "#051e18", class: "bg-[#051e18] border-emerald-950 text-white" },
  { name: "Charcoal Slate", value: "#1e293b", class: "bg-[#1e293b] border-slate-700 text-white" },
  { name: "Midnight Rose", value: "#1f1118", class: "bg-[#1f1118] border-rose-950 text-white" },
];

export default function WallpaperSettings({
  currentWallpaper,
  onWallpaperChange,
  showPatterns,
  onTogglePatterns
}: WallpaperSettingsProps) {
  const [imageUrl, setImageUrl] = useState("");
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleSelectSolid = (color: string) => {
    onWallpaperChange({ type: "solid", value: color });
  };

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageUrl.trim()) return;
    onWallpaperChange({ type: "custom", value: imageUrl.trim() });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result && typeof event.target.result === "string") {
        onWallpaperChange({ type: "custom", value: event.target.result });
      }
    };
    reader.readAsDataURL(file);
  };

  const handleReset = () => {
    onWallpaperChange({ type: "preset", value: "#efeae2" });
    setImageUrl("");
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-5 font-sans">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <h3 className="font-bold text-sm text-slate-800 flex items-center">
          <Palette className="w-4 h-4 text-[#00a884] mr-1.5" />
          <span>Chat Wallpaper Settings</span>
        </h3>
        <button
          onClick={handleReset}
          className="text-xs text-rose-500 hover:text-rose-600 font-semibold flex items-center space-x-1 hover:bg-rose-50 px-2 py-1 rounded transition"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Reset Default</span>
        </button>
      </div>

      {/* Preset solid colors selection */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-600 block">
          Solid Background Colors:
        </label>
        <div className="grid grid-cols-3 gap-2">
          {PRESET_COLORS.map((color) => {
            const isSelected = 
              (currentWallpaper.type === "solid" || currentWallpaper.type === "preset") && 
              currentWallpaper.value.toLowerCase() === color.value.toLowerCase();
            
            return (
              <button
                key={color.value}
                onClick={() => handleSelectSolid(color.value)}
                className={`h-11 rounded-lg border-2 text-left p-1.5 flex flex-col justify-between transition-all relative overflow-hidden group shadow-sm hover:scale-[1.02] cursor-pointer ${color.class} ${
                  isSelected ? "border-emerald-500 ring-2 ring-emerald-500/20" : "border-slate-200/60"
                }`}
                title={color.name}
              >
                <span className="text-[9px] font-medium truncate w-full opacity-0 group-hover:opacity-100 transition-opacity">
                  {color.name}
                </span>
                {isSelected && (
                  <div className="absolute inset-0 bg-black/5 flex items-center justify-center">
                    <Check className={`w-4 h-4 ${color.value === "#f8fafc" || color.value === "#efeae2" || color.value === "#cdeae3" || color.value === "#eae3f2" || color.value === "#fbe4db" ? "text-emerald-600" : "text-white"}`} />
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Doodle Pattern Overlay Switcher */}
      <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 flex items-center justify-between">
        <div className="space-y-0.5">
          <span className="text-xs font-bold text-slate-700 flex items-center">
            <Sparkles className="w-3.5 h-3.5 text-amber-500 mr-1" />
            WhatsApp Doodle Pattern
          </span>
          <p className="text-[10px] text-slate-500">
            Show the classic WhatsApp textured design overlay
          </p>
        </div>
        <button
          onClick={onTogglePatterns}
          className={`px-3 py-1 rounded-full text-xs font-bold transition-all cursor-pointer ${
            showPatterns 
              ? "bg-[#00a884] text-white shadow-sm" 
              : "bg-slate-200 text-slate-600"
          }`}
        >
          {showPatterns ? "ON" : "OFF"}
        </button>
      </div>

      {/* Custom wallpaper uploader */}
      <div className="space-y-3.5 border-t border-slate-100 pt-4">
        <label className="text-xs font-semibold text-slate-600 block flex items-center">
          <Image className="w-3.5 h-3.5 mr-1 text-[#128c7e]" />
          Custom Wallpaper Image:
        </label>

        {/* Upload via file */}
        <div className="flex space-x-2">
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 py-2 px-3 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-xs rounded-lg border border-slate-200 flex items-center justify-center space-x-1.5 transition cursor-pointer"
          >
            <Upload className="w-4 h-4 text-slate-500" />
            <span>Upload From Laptop</span>
          </button>
        </div>

        {/* Custom URL Input */}
        <form onSubmit={handleUrlSubmit} className="flex space-x-1.5">
          <div className="flex-1 flex items-center bg-slate-50 rounded-lg border border-slate-200/80 px-2.5 py-1">
            <Link className="w-3.5 h-3.5 text-slate-400 mr-1.5 shrink-0" />
            <input
              type="url"
              placeholder="Paste background image URL..."
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              className="w-full text-xs text-slate-700 bg-transparent focus:outline-none placeholder:text-slate-400"
            />
          </div>
          <button
            type="submit"
            className="py-1.5 px-3 bg-[#128c7e] hover:bg-[#075e54] text-white font-bold text-xs rounded-lg transition shrink-0"
          >
            Apply
          </button>
        </form>
      </div>

      {/* Live Active Wallpaper Preview */}
      <div className="border border-slate-200 rounded-xl overflow-hidden shadow-inner bg-slate-50">
        <div className="text-[10px] font-bold text-slate-400 bg-slate-100 py-1 px-3 border-b border-slate-200">
          LIVE PREVIEW (MINI)
        </div>
        <div className="p-4 flex justify-center">
          <div 
            className="w-32 h-20 rounded-lg border border-slate-300 relative shadow-md overflow-hidden flex items-center justify-center transition-all"
            style={{
              backgroundColor: currentWallpaper.type === "solid" || currentWallpaper.type === "preset" ? currentWallpaper.value : undefined,
              backgroundImage: currentWallpaper.type === "custom" ? `url(${currentWallpaper.value})` : undefined,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          >
            {/* Pattern */}
            {showPatterns && (currentWallpaper.type === "solid" || currentWallpaper.type === "preset") && (
              <div className="absolute inset-0 opacity-12 bg-[radial-gradient(#000000_10%,transparent_11%)] bg-[length:10px_10px]" />
            )}
            
            {/* Mock message bubble */}
            <div className="bg-[#d9fdd3] text-[#111b21] px-2 py-1 rounded-md text-[8px] font-sans shadow-sm absolute bottom-2 right-2">
              Looks awesome! 🍀
            </div>
            <div className="bg-white text-[#111b21] px-2 py-1 rounded-md text-[8px] font-sans shadow-sm absolute top-2 left-2">
              AI Bot active...
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
