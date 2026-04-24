const ParticlesJS = (function() {
    let scene, camera, renderer, particlesMesh, gridHelper, currentMode;
    let sphereColors = [];
    let animationFrameId;
    let mouseX = 0;
    let mouseY = 0;
    
    // Light Organic Minimalist Colors
    const sageGreen = new THREE.Color('#7AB88B');
    const primaryTeal = new THREE.Color('#84B4B9');
    const warningTerra = new THREE.Color('#E07A5F');
    const lightGray = new THREE.Color('#EAEAEA');
    
    function initParticleBackground(config) {
        const { containerId, particleCount = 800, mode = 'ambient', cameraZ = 5 } = config;
        currentMode = mode;
        
        const container = document.getElementById(containerId);
        if (!container) return null;
        
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = cameraZ;
        
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x7AB88B, 0.8);
        pointLight.position.set(5, 5, 5);
        scene.add(pointLight);
        
        const pointLight2 = new THREE.PointLight(0x84B4B9, 0.5);
        pointLight2.position.set(-5, -5, -5);
        scene.add(pointLight2);
        
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            if (mode === 'ambient') {
                positions[i * 3] = (Math.random() - 0.5) * 20;
                positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
                
                // Mix of Teal and Sage
                const c = Math.random() > 0.5 ? sageGreen : primaryTeal;
                colors[i * 3] = c.r;
                colors[i * 3 + 1] = c.g;
                colors[i * 3 + 2] = c.b;
            } else if (mode === 'sphere') {
                const phi = Math.acos(2 * Math.random() - 1);
                const theta = 2 * Math.PI * Math.random();
                const radius = 2.2 + (Math.random() * 0.1); 
                
                positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
                positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
                positions[i * 3 + 2] = radius * Math.cos(phi);
                
                // Sphere uses a uniform teal that can shift
                colors[i * 3] = primaryTeal.r;
                colors[i * 3 + 1] = primaryTeal.g;
                colors[i * 3 + 2] = primaryTeal.b;
                sphereColors.push(i);
            }
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const materialSize = mode === 'sphere' ? 0.03 : 0.02;
        const material = new THREE.PointsMaterial({
            size: materialSize,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            blending: THREE.NormalBlending
        });
        
        particlesMesh = new THREE.Points(geometry, material);
        scene.add(particlesMesh);
        
        if (mode === 'grid') {
            const gridGeom = new THREE.PlaneGeometry(100, 100, 40, 40);
            const gridMat = new THREE.LineBasicMaterial({ color: 0xDDDDDD, transparent: true, opacity: 0.5 });
            gridHelper = new THREE.LineSegments(
                new THREE.EdgesGeometry(gridGeom),
                gridMat
            );
            gridHelper.rotation.x = Math.PI / 2;
            gridHelper.position.y = -5;
            scene.add(gridHelper);
            camera.position.z = 15;
            particlesMesh.visible = false;
        }
        
        window.addEventListener('resize', onWindowResize);
        document.addEventListener('mousemove', onDocumentMouseMove);
        animate();
        
        return { scene, camera, renderer, updateSphereRisk };
    }
    
    function onDocumentMouseMove(event) {
        mouseX = (event.clientX - window.innerWidth / 2) * 0.002;
        mouseY = (event.clientY - window.innerHeight / 2) * 0.002;
    }
    
    function updateSphereRisk(riskValue) {
        if (currentMode !== 'sphere' || !particlesMesh) return;
        
        let targetColor = new THREE.Color();
        if (riskValue < 0.5) {
            targetColor.lerpColors(primaryTeal, sageGreen, riskValue * 2);
        } else {
            targetColor.lerpColors(sageGreen, warningTerra, (riskValue - 0.5) * 2);
        }
        
        const colors = particlesMesh.geometry.attributes.color.array;
        for (let i = 0; i < sphereColors.length; i++) {
            colors[i * 3] = targetColor.r;
            colors[i * 3 + 1] = targetColor.g;
            colors[i * 3 + 2] = targetColor.b;
        }
        particlesMesh.geometry.attributes.color.needsUpdate = true;
    }
    
    function onWindowResize() {
        if (camera && renderer) {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }
    
    function animate() {
        animationFrameId = requestAnimationFrame(animate);
        
        if (currentMode === 'ambient' && particlesMesh) {
            const positions = particlesMesh.geometry.attributes.position.array;
            for (let i = 0; i < positions.length / 3; i++) {
                positions[i * 3 + 1] += 0.01;
                positions[i * 3] += Math.sin(Date.now() * 0.001 + i) * 0.01;
                if (positions[i * 3 + 1] > 10) {
                    positions[i * 3 + 1] = -10;
                }
            }
            particlesMesh.geometry.attributes.position.needsUpdate = true;
        } else if (currentMode === 'sphere' && particlesMesh) {
            particlesMesh.rotation.y += 0.003;
            particlesMesh.rotation.x += 0.001;
        } else if (currentMode === 'grid' && gridHelper) {
            gridHelper.rotation.z += 0.001;
        }
        
        if (currentMode === 'ambient') {
            camera.position.x += (mouseX * 5 - camera.position.x) * 0.02;
            camera.position.y += (-mouseY * 5 - camera.position.y) * 0.02;
            camera.lookAt(scene.position);
        }
        
        renderer.render(scene, camera);
    }
    
    return {
        initParticleBackground,
        updateSphereRisk
    };
})();
