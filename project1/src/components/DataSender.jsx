import { useState } from 'react'
import { useWebSocket } from '../context/WebSocketContext'
import '../css/DataSender.css'

function DataSender() {
  const [inputMessage, setInputMessage] = useState('')
  const [error, setError] = useState('')
  const { messages, sendMessage, isConnected } = useWebSocket()

  const handleSubmit = (e) => {
    e.preventDefault()
    setError('')
    
    if (inputMessage.trim()) {
      try {
        const data = JSON.parse(inputMessage)
        if (!data.t) {
          data.t = Date.now()
        }
        sendMessage(JSON.stringify(data))
        setInputMessage('')
      } catch (err) {
        setError('Invalid JSON format')
      }
    }
  }

  // Get last sent and received messages
  const lastSent = messages.filter(m => m.type === 'sent').slice(-1)[0]
  const lastReceived = messages.filter(m => m.type === 'received').slice(-1)[0]

  return (
    <div className="data-sender-container">
      <div className="header">
        <span className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Connected' : '● Disconnected'}
        </span>
      </div>

      <form onSubmit={handleSubmit} className="json-form">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder='{"t": 1234567890, "u_ls": [45, 90], "tr_ls": [50, 95], ...}'
          rows="4"
          className={`json-textarea ${error ? 'error' : ''}`}
        />
        {error && <p className="error-message">{error}</p>}
        <button type="submit" disabled={!isConnected} className="submit-button">
          Send Data
        </button>
      </form>

      <div className="last-messages">
        {lastSent && (
          <div className="last-message">
            <span className="message-label">Last Sent:</span>
            <span className="message-preview">{lastSent.text.substring(0, 50)}...</span>
          </div>
        )}
        {lastReceived && (
          <div className="last-message">
            <span className="message-label">Last Received:</span>
            <span className="message-preview">{lastReceived.text.substring(0, 50)}...</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default DataSender
