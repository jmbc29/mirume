import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import PrivacyApp from './PrivacyApp'
import './App.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PrivacyApp />
  </StrictMode>,
)
