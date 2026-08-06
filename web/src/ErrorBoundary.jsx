import { Component } from "react";

const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("TradingTerminalNotebook crashed:", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className={`h-screen flex flex-col items-center justify-center gap-3 bg-[#121214] text-[#f0ede5] ${MONO} px-6 text-center`}>
        <div className="text-[#d8622b] text-sm font-bold tracking-wide uppercase">頁面發生錯誤</div>
        <div className="text-[12px] text-[#8d8d93] max-w-md leading-relaxed">
          {this.state.error.message || "未知錯誤"}
        </div>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-2 px-4 py-2 rounded-md border border-[rgba(201,161,92,.35)] bg-[rgba(201,161,92,.08)] text-[#c9a15c] text-[12px] font-semibold hover:bg-[rgba(201,161,92,.18)] transition-colors"
        >
          重新整理頁面
        </button>
      </div>
    );
  }
}
