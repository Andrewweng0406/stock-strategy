import React from "react";
import ReactDOM from "react-dom/client";
import TradingTerminalNotebook from "./TradingTerminalNotebook.jsx";
import ErrorBoundary from "./ErrorBoundary.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <TradingTerminalNotebook />
    </ErrorBoundary>
  </React.StrictMode>
);
