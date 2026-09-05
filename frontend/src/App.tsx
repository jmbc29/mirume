import { useMouseTracker } from "./hooks/useMouseTracker";
import HoverCard from "./components/HoverCard";
import "./App.css";

function App() {
  const { data, triggerPoint } = useMouseTracker();

  return (
    <div className="overlay" style={{ pointerEvents: "none" }}>
      <HoverCard data={data} triggerPoint={triggerPoint} />
    </div>
  );
}

export default App;
