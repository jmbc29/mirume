import { useMouseTracker } from "./hooks/useMouseTracker";
import HoverCard from "./components/HoverCard";
import "./App.css";

function App() {
  const { data, triggerPoint, setPaused } = useMouseTracker();

  return (
    <div className="overlay" style={{ pointerEvents: "none" }}>
      <HoverCard data={data} triggerPoint={triggerPoint} setHoverPaused={setPaused} />
    </div>
  );
}

export default App;
