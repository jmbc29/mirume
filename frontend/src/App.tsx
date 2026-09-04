import { useMouseTracker } from "./hooks/useMouseTracker";
import HoverCard from "./components/HoverCard";
import "./App.css";

function App() {
  const { data, cursor } = useMouseTracker();

  return (
    <div className="overlay" style={{ pointerEvents: "none" }}>
      <HoverCard data={data} cursor={cursor} />
    </div>
  );
}

export default App;
