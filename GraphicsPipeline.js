function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Rasterization Demo';
	UI.titleShort = 'rasterizationDemo';

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Rasterization`,
		id: `RasterizationDemoFS`,
		initialValue: `#define PROJECTION
#define RASTERIZATION
#define CLIPPING
#define INTERPOLATION
#define ZBUFFERING

precision highp float;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 VIEWPORT;

struct Vertex {
    vec3 position;
    vec3 color;
};

struct Polygon {
    // Numbers of vertices, i.e., points in the polygon
    int vertexCount;
    // The vertices themselves
    Vertex vertices[MAX_VERTEX_COUNT];
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) {
            polygon.vertices[i] = element;
        }
    }
    polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        destination.vertices[i] = source.vertices[i];
    }
    destination.vertexCount = source.vertexCount;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
    if (index >= polygon.vertexCount) index -= polygon.vertexCount;										//Added support for cases when index goes above vertexCount, by reverting to first vertex
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == index) return polygon.vertices[i];
    }
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
  polygon.vertexCount = 0;
}

// Clipping part

int edge2(vec2 point, Vertex a, Vertex b) {																//This function is a copy of the "edge" function below, in order to make use of in Clipping
  	float soln = (b.position[0] - a.position[0]) * (point[1] - a.position[1]) - (b.position[1] - a.position[1]) * (point[0] - a.position[0]);
  	if (soln <= 0.0)
      return 0;
    return 1;
}

#define ENTERING 0																				//These global variables are defined in order to differentiate the types of lines wrt boundaries
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {						//This function returns the previously defined global variables, based on the crossing type
#ifdef CLIPPING
  	vec2 point1 = vec2(poli1.position[0],poli1.position[1]);									//point1 and point2 are the intitial and final points of the line segment being intersected
  	vec2 point2 = vec2(poli2.position[0],poli2.position[1]);
  	int point1Status = edge2(point1,wind1,wind2);												// If edge2 returns 1, the point is outside, otherwise inside
  	int point2Status = edge2(point2,wind1,wind2);
  	if(point1Status == 1 && point2Status == 1)
      return OUTSIDE;
  	else if(point1Status == 1 && point2Status == 0)
      return ENTERING;
  	else if(point1Status == 0 && point2Status == 1)
      return LEAVING;
  	else
      return INSIDE;
    // Put your code here (DONE)
#else
    return INSIDE;
#endif
}

// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef CLIPPING
  	float m1 = (b.position[1] - a.position[1]) / (b.position[0] - a.position[0]);				//m1 is slope of line 1 (AB), and m3 the slope of line 2 (CD)
  	float m3 = (d.position[1] - c.position[1]) / (d.position[0] - c.position[0]);
	if(b.position[0] - a.position[0] == 0.0){													//The case when line 1 is vertical (slope is infinity)
      float solnX = a.position[0];
      float solnY = c.position[1] + m3 * (solnX - c.position[0]);
      float solnZ = d.position[2] - (((d.position[0] - solnX) * (d.position[2] - c.position[2])) / (d.position[0] - c.position[0]));
      solnZ = 1.0 / solnZ;
    }
  	if(d.position[0] - c.position[0] == 0.0){													//The case when line 2 (clipping window) is vertical (slope is infinity)
      float solnX = c.position[0];
      float solnY = a.position[1] + m1 * (solnX - a.position[0]);
      float solnZ = b.position[2] - (((b.position[0] - solnX) * (b.position[2] - a.position[2])) / (b.position[0] - a.position[0]));
      solnZ = 1.0 / solnZ;
  	  Vertex result;
  	  result.position = vec3(solnX,solnY,solnZ);
  	  result.color = a.color;
  	  return result;
    }
  	float solnX = ((c.position[1] - m3 * c.position[0]) - (a.position[1] - m1 * a.position[0])) / (m1 - m3);
  	float solnY = m1 * (solnX - a.position[0]) + a.position[1];
  	float solnZ = b.position[2] - (((b.position[0] - solnX) * (b.position[2] - a.position[2])) / (b.position[0] - a.position[0]));
  	solnZ = 1.0 / solnZ;																		//The z coordinate is inversely proportional to make it perspectively correct
  	Vertex result;
  	result.position = vec3(solnX,solnY,solnZ);
  	result.color = a.color;
  	return result;
    // Put your code here (DONE)
#else
    return a;
#endif
}

void sutherlandHodgmanClip(Polygon unclipped, Polygon clipWindow, out Polygon result) {
    Polygon clipped;
    copyPolygon(clipped, unclipped);

    // Loop over the clip window
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i >= clipWindow.vertexCount) break;

        // Make a temporary copy of the current clipped polygon
        Polygon oldClipped;
        copyPolygon(oldClipped, clipped);

        // Set the clipped polygon to be empty
        makeEmptyPolygon(clipped);

        // Loop over the current clipped polygon
        for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
            if (j >= oldClipped.vertexCount) break;
            
            // Handle the j-th vertex of the clipped polygon. This should make use of the function 
            // intersect() to be implemented above.
#ifdef CLIPPING
          	
          	Vertex nextPolyVertex = getWrappedPolygonVertex(oldClipped,j+1);
          	Vertex nextWindVertex = getWrappedPolygonVertex(clipWindow,i+1);
          	int crossType = getCrossType(oldClipped.vertices[j],nextPolyVertex,clipWindow.vertices[i],nextWindVertex);
          	Vertex intersection = intersect2D(oldClipped.vertices[j],nextPolyVertex,clipWindow.vertices[i],nextWindVertex);;
          	if(crossType == 0){													//If the line is entering the clip window, only the intersection point, and the final point are considered
              appendVertexToPolygon(clipped, intersection);
              appendVertexToPolygon(clipped, nextPolyVertex);
            }
          	else if(crossType == 1){											//If the line is leaving the clip window, only the intersection point is considered
              appendVertexToPolygon(clipped, intersection);
            }
          	else if(crossType == 3)												//If the line is completely inside the clip window, the final point of the line is appended
              appendVertexToPolygon(clipped, nextPolyVertex);
          																		//If the line is completely outside the clip window, nothing is done
            // Put your code here (DONE)
#else
            appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
        }
    }

    // Copy the last version to the output
    copyPolygon(result, clipped);
}

// Rasterization and culling part

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point RasterizationResolution settings

//return OUTER_SIDE;

// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {										//Original definition of edge function. It returns 1 or 0 based on the side of the line at which the point lies
#ifdef RASTERIZATION
  	float soln = (b.position[0] - a.position[0]) * (point[1] - a.position[1]) - (b.position[1] - a.position[1]) * (point[0] - a.position[0]);
  	if (soln <= 0.0)															//soln is obtained by substituting the point values in the line equation.
      return INNER_SIDE;
    // Put your code here (DONE)
#endif
    return OUTER_SIDE;
}

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
    // Don't evaluate empty polygons
    if (polygon.vertexCount == 0)
      return false;
    // Check against each edge of the polygon
    bool rasterise = true;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#ifdef RASTERIZATION
        	if(edge(point,polygon.vertices[i],getWrappedPolygonVertex(polygon,i+1)) == 1)		//Checking is done, for the line formed by every pair of vertices of the polygon. If in any case,
                return false;																	//the point lies outside the line, then the point lies outside the complete polygon.
            // Put your code here (DONE)
#else
            rasterise = false;
#endif
        }
    }
    return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
    float pointSize = 0.008;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            if(length(polygon.vertices[i].position.xy - point) < pointSize) return true;
        }
    }
    return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {									//This function is used to find the area of a triangle, if the vertices are known.
    // https://en.wikipedia.org/wiki/Heron%27s_formula
    float ab = length(a - b);
    float bc = length(b - c);
    float ca = length(c - a);
    float s = (ab + bc + ca) / 2.0;
    return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
    float weightSum = 0.0;														//weightSum is used to find the total area of the polygon
    vec3 colorSum = vec3(0.0);													//colorSum contains three values for the RGB components of color
    vec3 positionSum = vec3(0.0);												//positionSum contains three values for the XYZ components of positional coordinates.
    float depthSum = 0.0;														//depthSum is taken as the inverse of the z component, in order to make it perspectively correct in interpolation.
  	float weight;																//weight is the ratio to be taken, unique to every iteration.
    for (int i = 0; i < 3; ++i) {
        if (i < polygon.vertexCount) {
#if defined(INTERPOLATION) || defined(ZBUFFERING)
        	vec3 currColor = polygon.vertices[i].color;
          	vec3 currPosition = polygon.vertices[i].position;
          	Vertex nextVertex1 = getWrappedPolygonVertex(polygon, i+1);
          	Vertex nextVertex2 = getWrappedPolygonVertex(polygon, i+2);
          	vec2 pointB = vec2(nextVertex1.position[0],nextVertex1.position[1]);
          	vec2 pointC = vec2(nextVertex2.position[0],nextVertex2.position[1]);
          	weight = triangleArea(point,pointB,pointC);
          	weightSum = weightSum + weight;										//weightSum is found iteratively by adding the individual weights
            // Put your code here (DONE)
#else
#endif
#ifdef ZBUFFERING
          	positionSum = positionSum + (weight) * (currPosition);
          	depthSum = depthSum + (weight) / (currPosition[2]);					//Getting the depth ratio in each iteration
            // Put your code here (DONE)
#endif
#ifdef INTERPOLATION
          	colorSum = colorSum + (weight) * (currColor);						//Getting the color ratio in each iteration					
          	
            // Put your code here (DONE)
#endif
        }
    }
    
    Vertex result = polygon.vertices[0];										//Creating a new polygon to return as the result
  
#ifdef INTERPOLATION
    colorSum /= weightSum;														//Dividing by weightSum to get the final ratio
    depthSum /= weightSum;
  	positionSum /= weightSum;
    result.color = colorSum;
#endif
#ifdef ZBUFFERING
    positionSum[2] = positionSum[2] / depthSum;											//Reverting the inverse depthSum back to the required z coordinate.
    result.position = positionSum;
#endif

  return result;
}

// Projection part

// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
    mat4 projectionMatrix = mat4(1);

#ifdef PROJECTION
  	projectionMatrix[0] = vec4(1.0, 0.0, 0.0, 0.0);
	projectionMatrix[1] = vec4(0.0, 1.0, 0.0, 0.0);
	projectionMatrix[2] = vec4(0.0, 0.0, 1.0, 1.0);
	projectionMatrix[3] = vec4(0.0, 0.0, 0.0, 1.0);								//Taking an initial projection matrix
#endif
  
    return projectionMatrix;
}

// Used to generate a simple "look-at" camera. 
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    mat4 viewMatrix = mat4(1);

#ifdef PROJECTION
    vec3 VPN = TP - VRP;
  	// Generate the camera axes.
    vec3 n = normalize(VPN);
    vec3 u = normalize(cross(VUV, n));
    vec3 v = normalize(cross(n, u));
    
	viewMatrix[0] = vec4(u[0], v[0], n[0], 0);
	viewMatrix[1] = vec4(u[1], v[1], n[1], 0);
	viewMatrix[2] = vec4(u[2], v[2], n[2], 0);									//Taking view matrix, based on camera position and orientation.
	viewMatrix[3] = vec4(- dot(VRP, u), - dot(VRP, v), - dot(VRP, n), 1);
#endif
    return viewMatrix;
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec3 projectVertexPosition(vec3 position) {

  // Set the parameters for the look-at camera.
    vec3 TP = vec3(0, 0, 0);													//Parameters of the camera, specifying the location, normal, and the look-at direction
    vec3 VRP = vec3(0, 0, -7);
    vec3 VUV = vec3(0, 1, 0);
  
    // Compute the view matrix.
    mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  // Compute the projection matrix.
    mat4 projectionMatrix = computeProjectionMatrix();
  
#ifdef PROJECTION
    // Put your code here (DONE)
  	vec4 position2 = projectionMatrix * viewMatrix * vec4(position, 1.0);		//Used to convert object coordinates to world coordinates, and then into camera coordinates.
  	position = vec3(position2[0] / position2[3],position2[1] / position2[3], position2[2] / position2[3]);		//Converting the 4-element vector into cartesian 3-element vetor coordinates
  	return position;
#else
    
#endif
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
    copyPolygon(projectedPolygon, polygon);
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
        }
    }
}

// Draws a polygon by projecting, clipping, rasterizing and interpolating it
void drawPolygon(
  vec2 point, 
  Polygon clipWindow, 
  Polygon oldPolygon, 
  inout vec3 color, 
  inout float depth)
{
    Polygon projectedPolygon;
    projectPolygon(projectedPolygon, oldPolygon);  
  
    Polygon clippedPolygon;
    sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

    if (isPointInPolygon(point, clippedPolygon)) {
      
        Vertex interpolatedVertex = 
          interpolateVertex(point, projectedPolygon);
          
        if (interpolatedVertex.position.z < depth) {
            color = interpolatedVertex.color;
            depth = interpolatedVertex.position.z;
        }
    } else {
        if (isPointInPolygon(point, projectedPolygon)) {
            color = vec3(0.1, 0.1, 0.1);
        }
    }
  
   if (isPointOnPolygonVertex(point, clippedPolygon)) {
        color = vec3(1);
   }
}

// Main function calls

void drawScene(vec2 point, inout vec3 color) {
    color = vec3(0.3, 0.3, 0.3);
    point = vec2(2.0 * point.x / float(VIEWPORT.x) - 1.0, 2.0 * point.y / float(VIEWPORT.y) - 1.0);

    Polygon clipWindow;
    clipWindow.vertices[0].position = vec3(-0.750,  0.750, 1.0);
    clipWindow.vertices[1].position = vec3( 0.750,  0.750, 1.0);
    clipWindow.vertices[2].position = vec3( 0.750, -0.750, 1.0);
    clipWindow.vertices[3].position = vec3(-0.750, -0.750, 1.0);
    clipWindow.vertexCount = 4;
    color = isPointInPolygon(point, clipWindow) ? vec3(0.5, 0.5, 0.5) : color;

    const int triangleCount = 2;
    Polygon triangles[triangleCount];
  
    triangles[0].vertices[0].position = vec3(-7.7143, -3.8571, 1.0);
    triangles[0].vertices[1].position = vec3(7.7143, 8.4857, 1.0);
    triangles[0].vertices[2].position = vec3(4.8857, -0.5143, 1.0);
    triangles[0].vertices[0].color = vec3(1.0, 0.5, 0.1);
    triangles[0].vertices[1].color = vec3(0.2, 0.8, 0.2);
    triangles[0].vertices[2].color = vec3(0.2, 0.3, 1.0);
    triangles[0].vertexCount = 3;
  
    triangles[1].vertices[0].position = vec3(3.0836, -4.3820, 1.9);
    triangles[1].vertices[1].position = vec3(-3.9667, 0.7933, 0.5);
    triangles[1].vertices[2].position = vec3(-4.3714, 8.2286, 1.0);
    triangles[1].vertices[1].color = vec3(0.1, 0.5, 1.0);
    triangles[1].vertices[2].color = vec3(1.0, 0.6, 0.1);
    triangles[1].vertices[0].color = vec3(0.2, 0.6, 1.0);
    triangles[1].vertexCount = 3;

    float depth = 10000.0;
    // Project and draw all the triangles
    for (int i = 0; i < triangleCount; i++) {
        drawPolygon(point, clipWindow, triangles[i], color, depth);
    }   
}

void main() {
    drawScene(gl_FragCoord.xy, gl_FragColor.rgb);
    gl_FragColor.a = 1.0;
}`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: true,
		type: `text/javascript`,
		title: `Resolution settings`,
		id: `ResolutionJS`,
		initialValue: `// This variable sets the inverse scaling factor at which the rendering happens.
// The higher the constant, the faster it will be. SCALING = 1 is regular, non-scaled rendering.
SCALING = 1;`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoTextureVS - GL`,
		id: `RasterizationDemoTextureVS`,
		initialValue: `attribute vec3 position;
    attribute vec2 textureCoord;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    varying highp vec2 vTextureCoord;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        vTextureCoord = textureCoord;
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RasterizationDemoVS - GL`,
		id: `RasterizationDemoVS`,
		initialValue: `attribute vec3 position;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-fragment`,
		title: `RasterizationDemoTextureFS - GL`,
		id: `RasterizationDemoTextureFS`,
		initialValue: `
        varying highp vec2 vTextureCoord;

        uniform sampler2D uSampler;

        void main(void) {
            gl_FragColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
        }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
    try {
        gl = canvas.getContext("webgl");
        gl.viewportWidth = canvas.width;
        gl.viewportHeight = canvas.height;
    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(");
    }
}

function evalJS(id) {
    var jsScript = document.getElementById(id);
    eval(jsScript.innerHTML);
}

function getShader(gl, id) {
    var shaderScript = document.getElementById(id);
    if (!shaderScript) {
        return null;
    }

    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) {
            str += k.textContent;
        }
        k = k.nextSibling;
    }

    var shader;
    if (shaderScript.type == "x-shader/x-fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } else if (shaderScript.type == "x-shader/x-vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function RasterizationDemo() {
}

RasterizationDemo.prototype.initShaders = function() {

    this.shaderProgram = gl.createProgram();

    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoVS"));
    gl.attachShader(this.shaderProgram, getShader(gl, "RasterizationDemoFS"));
    gl.linkProgram(this.shaderProgram);

    if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.shaderProgram);

    this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
    gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

    this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
    this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initTextureShaders = function() {

    this.textureShaderProgram = gl.createProgram();

    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureVS"));
    gl.attachShader(this.textureShaderProgram, getShader(gl, "RasterizationDemoTextureFS"));
    gl.linkProgram(this.textureShaderProgram);

    if (!gl.getProgramParameter(this.textureShaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }

    gl.useProgram(this.textureShaderProgram);

    this.textureShaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.textureShaderProgram, "position");
    gl.enableVertexAttribArray(this.textureShaderProgram.vertexPositionAttribute);

    this.textureShaderProgram.textureCoordAttribute = gl.getAttribLocation(this.textureShaderProgram, "textureCoord");
    gl.enableVertexAttribArray(this.textureShaderProgram.textureCoordAttribute);
    //gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, 2, gl.FLOAT, false, 0, 0);

    this.textureShaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "projectionMatrix");
    this.textureShaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.textureShaderProgram, "modelViewMatrix");
}

RasterizationDemo.prototype.initBuffers = function() {
    this.triangleVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
    
    var vertices = [
         -1.0,  -1.0,  0.0,
         -1.0,   1.0,  0.0,
          1.0,   1.0,  0.0,

         -1.0,  -1.0,  0.0,
          1.0,  -1.0,  0.0,
          1.0,   1.0,  0.0,
     ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    this.triangleVertexPositionBuffer.itemSize = 3;
    this.triangleVertexPositionBuffer.numItems = 3 * 2;

    this.textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);

    var textureCoords = [
        0.0,  0.0,
        0.0,  1.0,
        1.0,  1.0,

        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoords), gl.STATIC_DRAW);
    this.textureCoordBuffer.itemSize = 2;
}

RasterizationDemo.prototype.initTextureFramebuffer = function() {
    // create off-screen framebuffer
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    this.framebuffer.width = this.prerender_width;
    this.framebuffer.height = this.prerender_height;

    // create RGB texture
    this.framebufferTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.framebuffer.width, this.framebuffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);//LINEAR_MIPMAP_NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    //gl.generateMipmap(gl.TEXTURE_2D);

    // create depth buffer
    this.renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.framebuffer.width, this.framebuffer.height);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.framebufferTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.renderbuffer);

    // reset state
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

RasterizationDemo.prototype.drawScene = function() {
            
    gl.bindFramebuffer(gl.FRAMEBUFFER, env.framebuffer);
    gl.useProgram(this.shaderProgram);
    gl.viewport(0, 0, this.prerender_width, this.prerender_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);

        gl.uniform2iv(gl.getUniformLocation(this.shaderProgram, "VIEWPORT"), [this.prerender_width, this.prerender_height]);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.textureShaderProgram);
    gl.viewport(0, 0, this.render_width, this.render_height);
    gl.clear(gl.COLOR_BUFFER_BIT);

        var perspectiveMatrix = new J3DIMatrix4();  
        perspectiveMatrix.setUniform(gl, this.textureShaderProgram.projectionMatrixUniform, false);

        var modelViewMatrix = new J3DIMatrix4();    
        modelViewMatrix.setUniform(gl, this.textureShaderProgram.modelviewMatrixUniform, false);

        gl.bindTexture(gl.TEXTURE_2D, this.framebufferTexture);
        gl.uniform1i(gl.getUniformLocation(this.textureShaderProgram, "uSampler"), 0);
            
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
        gl.vertexAttribPointer(this.textureShaderProgram.textureCoordAttribute, this.textureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);
        
        gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RasterizationDemo.prototype.run = function() {
    evalJS("ResolutionJS");

    this.render_width     = 800;
    this.render_height    = 400;

    this.prerender_width  = this.render_width / SCALING;
    this.prerender_height = this.render_height / SCALING;

    this.initTextureFramebuffer();
    this.initShaders();
    this.initTextureShaders();
    this.initBuffers();
};

function init() {   
    env = new RasterizationDemo();

    return env;
}

function compute(canvas)
{
    env.run();
    env.drawScene();
}
