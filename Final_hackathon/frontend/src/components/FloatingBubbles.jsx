import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const Bubble = ({ size, position, duration }) => {
  return (
    <motion.div
      className="absolute rounded-full bg-gradient-to-r from-indigo-100/10 to-purple-100/10 backdrop-blur-sm animate-float"
      style={{
        width: `${size}px`,
        height: `${size}px`,
        left: `${position.x}%`,
        top: `${position.y}%`,
        '--animation-duration': `${duration}s`
      }}
    />
  );
};

const FloatingBubbles = () => {
  const [bubbles, setBubbles] = useState([]);

  useEffect(() => {
    const bubbleCount = 8;
    const newBubbles = Array.from({ length: bubbleCount }, () => ({
      id: Math.random(),
      size: 40 + Math.random() * 60,
      position: {
        x: Math.random() * 100,
        y: Math.random() * 100
      },
      duration: 15 + Math.random() * 10
    }));
    setBubbles(newBubbles);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {bubbles.map(bubble => (
        <Bubble
          key={bubble.id}
          size={bubble.size}
          position={bubble.position}
          duration={bubble.duration}
        />
      ))}
    </div>
  );
};

export default FloatingBubbles;