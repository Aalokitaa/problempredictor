/**
 * KACHUA Particle System
 * Three.js based ambient, sphere, and grid particle effects
 * Globally accessible particle initialization and control
 */

let THREE = window.THREE || {};

function initParticleBackground(config = {}) {
  const {
    containerId = 'three-container',
    particleCount = 800,
    mode = 'ambient',
    cameraZ = 5
  } = config;

  const container = document.getElementById(containerId);
  if (!container && mode !== 'ambient') return null;

  // Scene setup
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0e0c09, 100, 200);

  const width = window.innerWidth;
  const height = window.innerHeight;

  // Camera
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.z = cameraZ;

  // Renderer
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(width, height);
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(window.devicePixelRatio);

  if (container) {
    container.appendChild(renderer.domElement);
  } else {
    document.body.appendChild(renderer.domElement);
    renderer.domElement.style.position = 'fixed';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.id = 'three-bg';
  }

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xfff5e0, 0.4);
  scene.add(ambientLight);

  const pointLight = new THREE.PointLight(0xd4a847, 0.8);
  pointLight.position.set(5, 5, 5);
  scene.add(pointLight);

  let particles, pointsMaterial, mesh;
  let updateFunction = null;

  // Ambient mode: drifting particles
  if (mode === 'ambient') {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 200;
      positions[i + 1] = (Math.random() - 0.5) * 200;
      positions[i + 2] = (Math.random() - 0.5) * 200;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    pointsMaterial = new THREE.PointsMaterial({
      color: 0xd4a847,
      size: 0.015,
      sizeAttenuation: true,
      opacity: 0.6
    });

    mesh = new THREE.Points(geometry, pointsMaterial);
    scene.add(mesh);

    let time = 0;
    updateFunction = () => {
      time += 0.001;
      const positions = geometry.attributes.position.array;

      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 1] += 0.1;
        positions[i] += Math.sin(time + i) * 0.01;

        if (positions[i + 1] > 100) {
          positions[i + 1] = -100;
        }
      }
      geometry.attributes.position.needsUpdate = true;
    };
  }

  // Sphere mode: rotating particles on sphere surface
  else if (mode === 'sphere') {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const radius = 2.2;

    for (let i = 0; i < particleCount; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const noise = (Math.random() - 0.5) * 0.3;

      positions[i * 3] = (radius + noise) * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = (radius + noise) * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = (radius + noise) * Math.cos(phi);

      // Color variation
      colors[i * 3] = 0.55;
      colors[i * 3 + 1] = 0.74;
      colors[i * 3 + 2] = 0.48;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    pointsMaterial = new THREE.PointsMaterial({
      size: 0.025,
      sizeAttenuation: true,
      vertexColors: true,
      opacity: 0.8
    });

    mesh = new THREE.Points(geometry, pointsMaterial);
    scene.add(mesh);

    updateFunction = (riskValue = 0) => {
      mesh.rotation.y += 0.003;
      mesh.rotation.x += 0.001;

      // Lerp material color based on risk
      if (riskValue !== undefined) {
        const safeColor = new THREE.Color(0x8fbc8f);
        const mediumColor = new THREE.Color(0xc4956a);
        const riskColor = new THREE.Color(0xb05c3a);

        let color;
        if (riskValue < 0.33) {
          color = safeColor.lerp(mediumColor, riskValue * 3);
        } else if (riskValue < 0.67) {
          color = mediumColor.lerp(riskColor, (riskValue - 0.33) * 3);
        } else {
          color = riskColor;
        }

        pointsMaterial.color = color;
      }
    };
  }

  // Grid mode: rotating grid background
  else if (mode === 'grid') {
    const gridSize = 50;
    const gridStep = 2;
    const geometry = new THREE.BufferGeometry();
    const positions = [];

    for (let x = -gridSize; x <= gridSize; x += gridStep) {
      for (let y = -gridSize; y <= gridSize; y += gridStep) {
        positions.push(x, y, 0);
        if (x < gridSize) positions.push(x + gridStep, y, 0);
        if (y < gridSize) positions.push(x, y + gridStep, 0);
      }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0xc4956a,
      opacity: 0.2,
      transparent: true
    });

    mesh = new THREE.LineSegments(geometry, lineMaterial);
    mesh.rotation.x = Math.PI * 0.3;
    scene.add(mesh);

    camera.position.z = 15;

    updateFunction = () => {
      mesh.rotation.z += 0.0005;
    };
  }

  // Resize handler
  const handleResize = () => {
    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);
  };

  window.addEventListener('resize', handleResize);

  // Animation loop
  const animate = () => {
    requestAnimationFrame(animate);
    if (updateFunction) updateFunction();
    renderer.render(scene, camera);
  };

  animate();

  // Return control object
  return {
    scene,
    camera,
    renderer,
    mesh,
    updateSphereRisk: (riskValue) => {
      if (mode === 'sphere' && updateFunction) {
        updateFunction(riskValue);
      }
    },
    updateFunction,
    dispose: () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    }
  };
}

// Global particle controller
window.particleController = {
  ambient: null,
  sphere: null,
  grid: null,

  initAmbient: (config = {}) => {
    window.particleController.ambient = initParticleBackground({
      ...config,
      mode: 'ambient',
      containerId: null
    });
    return window.particleController.ambient;
  },

  initSphere: (config = {}) => {
    window.particleController.sphere = initParticleBackground({
      ...config,
      mode: 'sphere',
      cameraZ: 5
    });
    return window.particleController.sphere;
  },

  initGrid: (config = {}) => {
    window.particleController.grid = initParticleBackground({
      ...config,
      mode: 'grid',
      cameraZ: 15
    });
    return window.particleController.grid;
  },

  updateSphereRisk: (riskValue) => {
    if (window.particleController.sphere) {
      window.particleController.sphere.updateSphereRisk(riskValue);
    }
  }
};

// Make available globally
window.initParticleBackground = initParticleBackground;
