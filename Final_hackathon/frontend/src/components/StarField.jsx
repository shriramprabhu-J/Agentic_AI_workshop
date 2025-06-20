import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const Star = ({ delay }) => {
  const randomX = Math.random() * 100;
  const randomDelay = delay + Math.random() * 2;
  const randomDuration = 3 + Math.random() * 2;
  const randomSize = 1 + Math.random() * 2;

  return (
    <motion.div
      className="absolute bg-white rounded-full animate-fall"
      style={{
        left: `${randomX}%`,
        width: `${randomSize}px`,
        height: `${randomSize}px`,
        animationDelay: `${randomDelay}s`,
        animationDuration: `${randomDuration}s`
      }}
    />
  );
};

const StarField = () => {
  const [stars, setStars] = useState([]);

  useEffect(() => {
    const starCount = 20;
    const newStars = Array.from({ length: starCount }, (_, i) => ({
      id: i,
      delay: i * 0.2
    }));
    setStars(newStars);

    const interval = setInterval(() => {
      setStars(prevStars => {
        const newStars = [...prevStars];
        newStars.push({
          id: Date.now(),
          delay: 0
        });
        if (newStars.length > starCount) {
          newStars.shift();
        }
        return newStars;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {stars.map(star => (
        <Star key={star.id} delay={star.delay} />
      ))}
    </div>
  );
};

export default StarField;