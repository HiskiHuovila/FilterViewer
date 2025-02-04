var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
/* =========================================================================
 *
 *  EgnType.ts
 *  static type
 *  v0.1
 *
 * ========================================================================= */
var EcognitaWeb3D;
(function (EcognitaWeb3D) {
    var Filter;
    (function (Filter) {
        Filter[Filter["LAPLACIAN"] = 0] = "LAPLACIAN";
        Filter[Filter["SOBEL"] = 1] = "SOBEL";
        Filter[Filter["GAUSSIAN"] = 2] = "GAUSSIAN";
        Filter[Filter["KUWAHARA"] = 3] = "KUWAHARA";
        Filter[Filter["GKUWAHARA"] = 4] = "GKUWAHARA";
        Filter[Filter["AKUWAHARA"] = 5] = "AKUWAHARA";
        Filter[Filter["ANISTROPIC"] = 6] = "ANISTROPIC";
        Filter[Filter["LIC"] = 7] = "LIC";
        Filter[Filter["NOISELIC"] = 8] = "NOISELIC";
        Filter[Filter["DoG"] = 9] = "DoG";
        Filter[Filter["XDoG"] = 10] = "XDoG";
        Filter[Filter["FDoG"] = 11] = "FDoG";
        Filter[Filter["FXDoG"] = 12] = "FXDoG";
        Filter[Filter["ABSTRACTION"] = 13] = "ABSTRACTION";
    })(Filter = EcognitaWeb3D.Filter || (EcognitaWeb3D.Filter = {}));
    var RenderPipeLine;
    (function (RenderPipeLine) {
        RenderPipeLine[RenderPipeLine["CONVOLUTION_FILTER"] = 0] = "CONVOLUTION_FILTER";
        RenderPipeLine[RenderPipeLine["ANISTROPIC"] = 1] = "ANISTROPIC";
        RenderPipeLine[RenderPipeLine["BLOOM_EFFECT"] = 2] = "BLOOM_EFFECT";
        RenderPipeLine[RenderPipeLine["CONVOLUTION_TWICE"] = 3] = "CONVOLUTION_TWICE";
        RenderPipeLine[RenderPipeLine["ABSTRACTION"] = 4] = "ABSTRACTION";
    })(RenderPipeLine = EcognitaWeb3D.RenderPipeLine || (EcognitaWeb3D.RenderPipeLine = {}));
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
/* =========================================================================
 *
 *  cv_imread.ts
 *  read the image file
 *
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    function imread(file) {
        var img = new Image();
        img.src = file;
        return img;
    }
    EcognitaMathLib.imread = imread;
})(EcognitaMathLib || (EcognitaMathLib = {}));
/* =========================================================================
 *
 *  cv_colorSpace.ts
 *  color space transformation
 *
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    //hsv space transform to rgb space
    //h(0~360) sva(0~1)
    function HSV2RGB(h, s, v, a) {
        if (s > 1 || v > 1 || a > 1) {
            return;
        }
        var th = h % 360;
        var i = Math.floor(th / 60);
        var f = th / 60 - i;
        var m = v * (1 - s);
        var n = v * (1 - s * f);
        var k = v * (1 - s * (1 - f));
        var color = new Array();
        if (!(s > 0) && !(s < 0)) {
            color.push(v, v, v, a);
        }
        else {
            var r = new Array(v, n, m, m, k, v);
            var g = new Array(k, v, v, n, m, m);
            var b = new Array(m, m, k, v, v, n);
            color.push(r[i], g[i], b[i], a);
        }
        return color;
    }
    EcognitaMathLib.HSV2RGB = HSV2RGB;
})(EcognitaMathLib || (EcognitaMathLib = {}));
/* =========================================================================
 *
 *  extra_utils.ts
 *  simple utils from extra library
 *
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    var Hammer_Utils = /** @class */ (function () {
        //event listener dom,
        function Hammer_Utils(canvas) {
            //event listener
            //using hammer library
            this.hm = new Hammer(canvas);
            this.on_pan = undefined;
        }
        Hammer_Utils.prototype.enablePan = function () {
            if (this.on_pan == undefined) {
                console.log("please setting the PAN function!");
                return;
            }
            this.hm.add(new Hammer.Pan({ direction: Hammer.DIRECTION_ALL, threshold: 0 }));
            this.hm.on("pan", this.on_pan);
        };
        return Hammer_Utils;
    }());
    EcognitaMathLib.Hammer_Utils = Hammer_Utils;
})(EcognitaMathLib || (EcognitaMathLib = {}));
/* =========================================================================
 *
 *  webgl_matrix.ts
 *  a matrix library developed for webgl
 *  part of source code referenced by minMatrix.js
 *  https://wgld.org/d/library/l001.html
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    var WebGLMatrix = /** @class */ (function () {
        function WebGLMatrix() {
            this.inverse = function (mat1) {
                var mat = this.create();
                var a = mat1[0], b = mat1[1], c = mat1[2], d = mat1[3], e = mat1[4], f = mat1[5], g = mat1[6], h = mat1[7], i = mat1[8], j = mat1[9], k = mat1[10], l = mat1[11], m = mat1[12], n = mat1[13], o = mat1[14], p = mat1[15], q = a * f - b * e, r = a * g - c * e, s = a * h - d * e, t = b * g - c * f, u = b * h - d * f, v = c * h - d * g, w = i * n - j * m, x = i * o - k * m, y = i * p - l * m, z = j * o - k * n, A = j * p - l * n, B = k * p - l * o, ivd = 1 / (q * B - r * A + s * z + t * y - u * x + v * w);
                mat[0] = (f * B - g * A + h * z) * ivd;
                mat[1] = (-b * B + c * A - d * z) * ivd;
                mat[2] = (n * v - o * u + p * t) * ivd;
                mat[3] = (-j * v + k * u - l * t) * ivd;
                mat[4] = (-e * B + g * y - h * x) * ivd;
                mat[5] = (a * B - c * y + d * x) * ivd;
                mat[6] = (-m * v + o * s - p * r) * ivd;
                mat[7] = (i * v - k * s + l * r) * ivd;
                mat[8] = (e * A - f * y + h * w) * ivd;
                mat[9] = (-a * A + b * y - d * w) * ivd;
                mat[10] = (m * u - n * s + p * q) * ivd;
                mat[11] = (-i * u + j * s - l * q) * ivd;
                mat[12] = (-e * z + f * x - g * w) * ivd;
                mat[13] = (a * z - b * x + c * w) * ivd;
                mat[14] = (-m * t + n * r - o * q) * ivd;
                mat[15] = (i * t - j * r + k * q) * ivd;
                return mat;
            };
        }
        WebGLMatrix.prototype.create = function () { return new Float32Array(16); };
        WebGLMatrix.prototype.identity = function (mat) {
            mat[0] = 1;
            mat[1] = 0;
            mat[2] = 0;
            mat[3] = 0;
            mat[4] = 0;
            mat[5] = 1;
            mat[6] = 0;
            mat[7] = 0;
            mat[8] = 0;
            mat[9] = 0;
            mat[10] = 1;
            mat[11] = 0;
            mat[12] = 0;
            mat[13] = 0;
            mat[14] = 0;
            mat[15] = 1;
            return mat;
        };
        //mat2 x mat1,give mat1 a transform(mat2)
        WebGLMatrix.prototype.multiply = function (mat1, mat2) {
            var mat = this.create();
            var a = mat1[0], b = mat1[1], c = mat1[2], d = mat1[3], e = mat1[4], f = mat1[5], g = mat1[6], h = mat1[7], i = mat1[8], j = mat1[9], k = mat1[10], l = mat1[11], m = mat1[12], n = mat1[13], o = mat1[14], p = mat1[15], A = mat2[0], B = mat2[1], C = mat2[2], D = mat2[3], E = mat2[4], F = mat2[5], G = mat2[6], H = mat2[7], I = mat2[8], J = mat2[9], K = mat2[10], L = mat2[11], M = mat2[12], N = mat2[13], O = mat2[14], P = mat2[15];
            mat[0] = A * a + B * e + C * i + D * m;
            mat[1] = A * b + B * f + C * j + D * n;
            mat[2] = A * c + B * g + C * k + D * o;
            mat[3] = A * d + B * h + C * l + D * p;
            mat[4] = E * a + F * e + G * i + H * m;
            mat[5] = E * b + F * f + G * j + H * n;
            mat[6] = E * c + F * g + G * k + H * o;
            mat[7] = E * d + F * h + G * l + H * p;
            mat[8] = I * a + J * e + K * i + L * m;
            mat[9] = I * b + J * f + K * j + L * n;
            mat[10] = I * c + J * g + K * k + L * o;
            mat[11] = I * d + J * h + K * l + L * p;
            mat[12] = M * a + N * e + O * i + P * m;
            mat[13] = M * b + N * f + O * j + P * n;
            mat[14] = M * c + N * g + O * k + P * o;
            mat[15] = M * d + N * h + O * l + P * p;
            return mat;
        };
        WebGLMatrix.prototype.scale = function (mat1, param_scale) {
            var mat = this.create();
            if (param_scale.length != 3)
                return undefined;
            mat[0] = mat1[0] * param_scale[0];
            mat[1] = mat1[1] * param_scale[0];
            mat[2] = mat1[2] * param_scale[0];
            mat[3] = mat1[3] * param_scale[0];
            mat[4] = mat1[4] * param_scale[1];
            mat[5] = mat1[5] * param_scale[1];
            mat[6] = mat1[6] * param_scale[1];
            mat[7] = mat1[7] * param_scale[1];
            mat[8] = mat1[8] * param_scale[2];
            mat[9] = mat1[9] * param_scale[2];
            mat[10] = mat1[10] * param_scale[2];
            mat[11] = mat1[11] * param_scale[2];
            mat[12] = mat1[12];
            mat[13] = mat1[13];
            mat[14] = mat1[14];
            mat[15] = mat1[15];
            return mat;
        };
        //vec * matrix,so translate matrix should use its transpose matrix
        WebGLMatrix.prototype.translate = function (mat1, param_translate) {
            var mat = this.create();
            if (param_translate.length != 3)
                return undefined;
            mat[0] = mat1[0];
            mat[1] = mat1[1];
            mat[2] = mat1[2];
            mat[3] = mat1[3];
            mat[4] = mat1[4];
            mat[5] = mat1[5];
            mat[6] = mat1[6];
            mat[7] = mat1[7];
            mat[8] = mat1[8];
            mat[9] = mat1[9];
            mat[10] = mat1[10];
            mat[11] = mat1[11];
            mat[12] = mat1[0] * param_translate[0] + mat1[4] * param_translate[1] + mat1[8] * param_translate[2] + mat1[12];
            mat[13] = mat1[1] * param_translate[0] + mat1[5] * param_translate[1] + mat1[9] * param_translate[2] + mat1[13];
            mat[14] = mat1[2] * param_translate[0] + mat1[6] * param_translate[1] + mat1[10] * param_translate[2] + mat1[14];
            mat[15] = mat1[3] * param_translate[0] + mat1[7] * param_translate[1] + mat1[11] * param_translate[2] + mat1[15];
            return mat;
        };
        // https://dspace.lboro.ac.uk/dspace-jspui/handle/2134/18050
        WebGLMatrix.prototype.rotate = function (mat1, angle, axis) {
            var mat = this.create();
            if (axis.length != 3)
                return undefined;
            var sq = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            if (!sq) {
                return undefined;
            }
            var a = axis[0], b = axis[1], c = axis[2];
            if (sq != 1) {
                sq = 1 / sq;
                a *= sq;
                b *= sq;
                c *= sq;
            }
            var d = Math.sin(angle), e = Math.cos(angle), f = 1 - e, g = mat1[0], h = mat1[1], i = mat1[2], j = mat1[3], k = mat1[4], l = mat1[5], m = mat1[6], n = mat1[7], o = mat1[8], p = mat1[9], q = mat1[10], r = mat1[11], s = a * a * f + e, t = b * a * f + c * d, u = c * a * f - b * d, v = a * b * f - c * d, w = b * b * f + e, x = c * b * f + a * d, y = a * c * f + b * d, z = b * c * f - a * d, A = c * c * f + e;
            if (angle) {
                if (mat1 != mat) {
                    mat[12] = mat1[12];
                    mat[13] = mat1[13];
                    mat[14] = mat1[14];
                    mat[15] = mat1[15];
                }
            }
            else {
                mat = mat1;
            }
            mat[0] = g * s + k * t + o * u;
            mat[1] = h * s + l * t + p * u;
            mat[2] = i * s + m * t + q * u;
            mat[3] = j * s + n * t + r * u;
            mat[4] = g * v + k * w + o * x;
            mat[5] = h * v + l * w + p * x;
            mat[6] = i * v + m * w + q * x;
            mat[7] = j * v + n * w + r * x;
            mat[8] = g * y + k * z + o * A;
            mat[9] = h * y + l * z + p * A;
            mat[10] = i * y + m * z + q * A;
            mat[11] = j * y + n * z + r * A;
            return mat;
        };
        WebGLMatrix.prototype.viewMatrix = function (cam, target, up) {
            var mat = this.identity(this.create());
            if (cam.length != 3 || target.length != 3 || up.length != 3)
                return undefined;
            var camX = cam[0], camY = cam[1], camZ = cam[2];
            var targetX = target[0], targetY = target[1], targetZ = target[2];
            var upX = up[0], upY = up[1], upZ = up[2];
            //cam and target have the same position
            if (camX == targetX && camY == targetY && camZ == targetZ)
                return mat;
            var forwardX = camX - targetX, forwardY = camY - targetY, forwardZ = camZ - targetZ;
            var l = 1 / Math.sqrt(forwardX * forwardX + forwardY * forwardY + forwardZ * forwardZ);
            forwardX *= l;
            forwardY *= l;
            forwardZ *= l;
            var rightX = upY * forwardZ - upZ * forwardY;
            var rightY = upZ * forwardX - upX * forwardZ;
            var rightZ = upX * forwardY - upY * forwardX;
            l = Math.sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
            if (!l) {
                rightX = 0;
                rightY = 0;
                rightZ = 0;
            }
            else {
                l = 1 / Math.sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
                rightX *= l;
                rightY *= l;
                rightZ *= l;
            }
            upX = forwardY * rightZ - forwardZ * rightY;
            upY = forwardZ * rightX - forwardX * rightZ;
            upZ = forwardX * rightY - forwardY * rightX;
            mat[0] = rightX;
            mat[1] = upX;
            mat[2] = forwardX;
            mat[3] = 0;
            mat[4] = rightY;
            mat[5] = upY;
            mat[6] = forwardY;
            mat[7] = 0;
            mat[8] = rightZ;
            mat[9] = upZ;
            mat[10] = forwardZ;
            mat[11] = 0;
            mat[12] = -(rightX * camX + rightY * camY + rightZ * camZ);
            mat[13] = -(upX * camX + upY * camY + upZ * camZ);
            mat[14] = -(forwardX * camX + forwardY * camY + forwardZ * camZ);
            mat[15] = 1;
            return mat;
        };
        WebGLMatrix.prototype.perspectiveMatrix = function (fovy, aspect, near, far) {
            var mat = this.identity(this.create());
            var t = near * Math.tan(fovy * Math.PI / 360);
            var r = t * aspect;
            var a = r * 2, b = t * 2, c = far - near;
            mat[0] = near * 2 / a;
            mat[1] = 0;
            mat[2] = 0;
            mat[3] = 0;
            mat[4] = 0;
            mat[5] = near * 2 / b;
            mat[6] = 0;
            mat[7] = 0;
            mat[8] = 0;
            mat[9] = 0;
            mat[10] = -(far + near) / c;
            mat[11] = -1;
            mat[12] = 0;
            mat[13] = 0;
            mat[14] = -(far * near * 2) / c;
            mat[15] = 0;
            return mat;
        };
        WebGLMatrix.prototype.orthoMatrix = function (left, right, top, bottom, near, far) {
            var mat = this.identity(this.create());
            var h = (right - left);
            var v = (top - bottom);
            var d = (far - near);
            mat[0] = 2 / h;
            mat[1] = 0;
            mat[2] = 0;
            mat[3] = 0;
            mat[4] = 0;
            mat[5] = 2 / v;
            mat[6] = 0;
            mat[7] = 0;
            mat[8] = 0;
            mat[9] = 0;
            mat[10] = -2 / d;
            mat[11] = 0;
            mat[12] = -(left + right) / h;
            mat[13] = -(top + bottom) / v;
            mat[14] = -(far + near) / d;
            mat[15] = 1;
            return mat;
        };
        ;
        WebGLMatrix.prototype.transpose = function (mat1) {
            var mat = this.create();
            mat[0] = mat1[0];
            mat[1] = mat1[4];
            mat[2] = mat1[8];
            mat[3] = mat1[12];
            mat[4] = mat1[1];
            mat[5] = mat1[5];
            mat[6] = mat1[9];
            mat[7] = mat1[13];
            mat[8] = mat1[2];
            mat[9] = mat1[6];
            mat[10] = mat1[10];
            mat[11] = mat1[14];
            mat[12] = mat1[3];
            mat[13] = mat1[7];
            mat[14] = mat1[11];
            mat[15] = mat1[15];
            return mat;
        };
        return WebGLMatrix;
    }());
    EcognitaMathLib.WebGLMatrix = WebGLMatrix;
})(EcognitaMathLib || (EcognitaMathLib = {}));
/* =========================================================================
 *
 *  webgl_quaternion.ts
 *  a quaternion library developed for webgl
 *  part of source code referenced by minMatrix.js
 *  https://wgld.org/d/library/l001.html
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    var WebGLQuaternion = /** @class */ (function () {
        function WebGLQuaternion() {
        }
        WebGLQuaternion.prototype.create = function () { return new Float32Array(4); };
        WebGLQuaternion.prototype.identity = function (qat) {
            qat[0] = 0;
            qat[1] = 0;
            qat[2] = 0;
            qat[3] = 1;
            return qat;
        };
        WebGLQuaternion.prototype.inverse = function (qat) {
            var out_qat = this.create();
            out_qat[0] = -qat[0];
            out_qat[1] = -qat[1];
            out_qat[2] = -qat[2];
            out_qat[3] = qat[3];
            return out_qat;
        };
        WebGLQuaternion.prototype.normalize = function (qat) {
            var x = qat[0], y = qat[1], z = qat[2], w = qat[3];
            var l = Math.sqrt(x * x + y * y + z * z + w * w);
            if (l === 0) {
                qat[0] = 0;
                qat[1] = 0;
                qat[2] = 0;
                qat[3] = 0;
            }
            else {
                l = 1 / l;
                qat[0] = x * l;
                qat[1] = y * l;
                qat[2] = z * l;
                qat[3] = w * l;
            }
            return qat;
        };
        //q1(v1,w1) q2(v2,w2)
        //v(Im) =  xi + yj + zk
        //w(Re)
        //q1q2 = (v1 x v2 + w2v1 + w1v2,w1w2 - v1・v2)
        WebGLQuaternion.prototype.multiply = function (qat1, qat2) {
            var out_qat = this.create();
            var ax = qat1[0], ay = qat1[1], az = qat1[2], aw = qat1[3];
            var bx = qat2[0], by = qat2[1], bz = qat2[2], bw = qat2[3];
            out_qat[0] = ax * bw + aw * bx + ay * bz - az * by;
            out_qat[1] = ay * bw + aw * by + az * bx - ax * bz;
            out_qat[2] = az * bw + aw * bz + ax * by - ay * bx;
            out_qat[3] = aw * bw - ax * bx - ay * by - az * bz;
            return out_qat;
        };
        WebGLQuaternion.prototype.rotate = function (angle, axis) {
            var sq = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            if (!sq) {
                console.log("need a axis value");
                return undefined;
            }
            var a = axis[0], b = axis[1], c = axis[2];
            if (sq != 1) {
                sq = 1 / sq;
                a *= sq;
                b *= sq;
                c *= sq;
            }
            var s = Math.sin(angle * 0.5);
            var out_qat = this.create();
            out_qat[0] = a * s;
            out_qat[1] = b * s;
            out_qat[2] = c * s;
            out_qat[3] = Math.cos(angle * 0.5);
            return out_qat;
        };
        //P' = qPq^(-1)
        WebGLQuaternion.prototype.ToV3 = function (p_v3, q) {
            var out_p = new Array(3);
            var inv_q = this.inverse(q);
            var in_p = this.create();
            in_p[0] = p_v3[0];
            in_p[1] = p_v3[1];
            in_p[2] = p_v3[2];
            var p_invq = this.multiply(inv_q, in_p);
            var q_p_invq = this.multiply(p_invq, q);
            out_p[0] = q_p_invq[0];
            out_p[1] = q_p_invq[1];
            out_p[2] = q_p_invq[2];
            return out_p;
        };
        WebGLQuaternion.prototype.ToMat4x4 = function (q) {
            var out_mat = new Float32Array(16);
            var x = q[0], y = q[1], z = q[2], w = q[3];
            var x2 = x + x, y2 = y + y, z2 = z + z;
            var xx = x * x2, xy = x * y2, xz = x * z2;
            var yy = y * y2, yz = y * z2, zz = z * z2;
            var wx = w * x2, wy = w * y2, wz = w * z2;
            out_mat[0] = 1 - (yy + zz);
            out_mat[1] = xy - wz;
            out_mat[2] = xz + wy;
            out_mat[3] = 0;
            out_mat[4] = xy + wz;
            out_mat[5] = 1 - (xx + zz);
            out_mat[6] = yz - wx;
            out_mat[7] = 0;
            out_mat[8] = xz - wy;
            out_mat[9] = yz + wx;
            out_mat[10] = 1 - (xx + yy);
            out_mat[11] = 0;
            out_mat[12] = 0;
            out_mat[13] = 0;
            out_mat[14] = 0;
            out_mat[15] = 1;
            return out_mat;
        };
        WebGLQuaternion.prototype.slerp = function (qtn1, qtn2, time) {
            if (time < 0 || time > 1) {
                console.log("parameter time's setting is wrong!");
                return undefined;
            }
            var out_q = this.create();
            var ht = qtn1[0] * qtn2[0] + qtn1[1] * qtn2[1] + qtn1[2] * qtn2[2] + qtn1[3] * qtn2[3];
            var hs = 1.0 - ht * ht;
            if (hs <= 0.0) {
                out_q[0] = qtn1[0];
                out_q[1] = qtn1[1];
                out_q[2] = qtn1[2];
                out_q[3] = qtn1[3];
            }
            else {
                hs = Math.sqrt(hs);
                if (Math.abs(hs) < 0.0001) {
                    out_q[0] = (qtn1[0] * 0.5 + qtn2[0] * 0.5);
                    out_q[1] = (qtn1[1] * 0.5 + qtn2[1] * 0.5);
                    out_q[2] = (qtn1[2] * 0.5 + qtn2[2] * 0.5);
                    out_q[3] = (qtn1[3] * 0.5 + qtn2[3] * 0.5);
                }
                else {
                    var ph = Math.acos(ht);
                    var pt = ph * time;
                    var t0 = Math.sin(ph - pt) / hs;
                    var t1 = Math.sin(pt) / hs;
                    out_q[0] = qtn1[0] * t0 + qtn2[0] * t1;
                    out_q[1] = qtn1[1] * t0 + qtn2[1] * t1;
                    out_q[2] = qtn1[2] * t0 + qtn2[2] * t1;
                    out_q[3] = qtn1[3] * t0 + qtn2[3] * t1;
                }
            }
            return out_q;
        };
        return WebGLQuaternion;
    }());
    EcognitaMathLib.WebGLQuaternion = WebGLQuaternion;
})(EcognitaMathLib || (EcognitaMathLib = {}));
/* =========================================================================
 *
 *  webgl_utils.ts
 *  utils for webgl
 *  part of source code referenced by tantalum-gl.js
 * ========================================================================= */
var EcognitaMathLib;
(function (EcognitaMathLib) {
    function GetGLTypeSize(type) {
        switch (type) {
            case gl.BYTE:
            case gl.UNSIGNED_BYTE:
                return 1;
            case gl.SHORT:
            case gl.UNSIGNED_SHORT:
                return 2;
            case gl.INT:
            case gl.UNSIGNED_INT:
            case gl.FLOAT:
                return 4;
            default:
                return 0;
        }
    }
    EcognitaMathLib.GetGLTypeSize = GetGLTypeSize;
    var WebGL_Texture = /** @class */ (function () {
        function WebGL_Texture(channels, isFloat, texels, texType, texInterpolation, useMipmap) {
            if (texType === void 0) { texType = gl.REPEAT; }
            if (texInterpolation === void 0) { texInterpolation = gl.LINEAR; }
            if (useMipmap === void 0) { useMipmap = true; }
            this.type = isFloat ? gl.FLOAT : gl.UNSIGNED_BYTE;
            this.format = [gl.LUMINANCE, gl.RG, gl.RGB, gl.RGBA][channels - 1];
            this.glName = gl.createTexture();
            this.bind(this.glName);
            gl.texImage2D(gl.TEXTURE_2D, 0, this.format, this.format, this.type, texels);
            if (useMipmap) {
                gl.generateMipmap(gl.TEXTURE_2D);
            }
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, texInterpolation);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, texInterpolation);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, texType);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, texType);
            this.texture = this.glName;
            this.bind(null);
        }
        WebGL_Texture.prototype.bind = function (tex) {
            gl.bindTexture(gl.TEXTURE_2D, tex);
        };
        return WebGL_Texture;
    }());
    EcognitaMathLib.WebGL_Texture = WebGL_Texture;
    var WebGL_CubeMapTexture = /** @class */ (function () {
        function WebGL_CubeMapTexture(texArray) {
            this.cubeSource = texArray;
            this.cubeTarget = new Array(gl.TEXTURE_CUBE_MAP_POSITIVE_X, gl.TEXTURE_CUBE_MAP_POSITIVE_Y, gl.TEXTURE_CUBE_MAP_POSITIVE_Z, gl.TEXTURE_CUBE_MAP_NEGATIVE_X, gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, gl.TEXTURE_CUBE_MAP_NEGATIVE_Z);
            this.loadCubeTexture();
            this.cubeTexture = undefined;
        }
        WebGL_CubeMapTexture.prototype.loadCubeTexture = function () {
            var _this = this;
            var cubeImage = new Array();
            var loadFlagCnt = 0;
            this.cubeImage = cubeImage;
            for (var i = 0; i < this.cubeSource.length; i++) {
                cubeImage[i] = new Object();
                cubeImage[i].data = new Image();
                cubeImage[i].data.src = this.cubeSource[i];
                cubeImage[i].data.onload = (function () {
                    loadFlagCnt++;
                    //check image load
                    if (loadFlagCnt == _this.cubeSource.length) {
                        _this.generateCubeMap();
                    }
                });
            }
        };
        WebGL_CubeMapTexture.prototype.generateCubeMap = function () {
            var tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
            for (var j = 0; j < this.cubeSource.length; j++) {
                gl.texImage2D(this.cubeTarget[j], 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.cubeImage[j].data);
            }
            gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this.cubeTexture = tex;
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
        };
        return WebGL_CubeMapTexture;
    }());
    EcognitaMathLib.WebGL_CubeMapTexture = WebGL_CubeMapTexture;
    var WebGL_RenderTarget = /** @class */ (function () {
        function WebGL_RenderTarget() {
            this.glName = gl.createFramebuffer();
        }
        WebGL_RenderTarget.prototype.bind = function () { gl.bindFramebuffer(gl.FRAMEBUFFER, this.glName); };
        WebGL_RenderTarget.prototype.unbind = function () { gl.bindFramebuffer(gl.FRAMEBUFFER, null); };
        WebGL_RenderTarget.prototype.attachTexture = function (texture, index) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + index, gl.TEXTURE_2D, texture.glName, 0);
        };
        WebGL_RenderTarget.prototype.detachTexture = function (index) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + index, gl.TEXTURE_2D, null, 0);
        };
        WebGL_RenderTarget.prototype.drawBuffers = function (numBufs) {
            var buffers = [];
            for (var i = 0; i < numBufs; ++i)
                buffers.push(gl.COLOR_ATTACHMENT0 + i);
            multiBufExt.drawBuffersWEBGL(buffers);
        };
        return WebGL_RenderTarget;
    }());
    EcognitaMathLib.WebGL_RenderTarget = WebGL_RenderTarget;
    var WebGL_Shader = /** @class */ (function () {
        function WebGL_Shader(shaderDict, vert, frag) {
            this.vertex = this.createShaderObject(shaderDict, vert, false);
            this.fragment = this.createShaderObject(shaderDict, frag, true);
            this.program = gl.createProgram();
            gl.attachShader(this.program, this.vertex);
            gl.attachShader(this.program, this.fragment);
            gl.linkProgram(this.program);
            this.uniforms = {};
            if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
                alert("Could not initialise shaders");
        }
        WebGL_Shader.prototype.bind = function () {
            gl.useProgram(this.program);
        };
        WebGL_Shader.prototype.createShaderObject = function (shaderDict, name, isFragment) {
            var shaderSource = this.resolveShaderSource(shaderDict, name);
            var shaderObject = gl.createShader(isFragment ? gl.FRAGMENT_SHADER : gl.VERTEX_SHADER);
            gl.shaderSource(shaderObject, shaderSource);
            gl.compileShader(shaderObject);
            if (!gl.getShaderParameter(shaderObject, gl.COMPILE_STATUS)) {
                /* Add some line numbers for convenience */
                var lines = shaderSource.split("\n");
                for (var i = 0; i < lines.length; ++i)
                    lines[i] = ("   " + (i + 1)).slice(-4) + " | " + lines[i];
                shaderSource = lines.join("\n");
                throw new Error((isFragment ? "Fragment" : "Vertex") + " shader compilation error for shader '" + name + "':\n\n    " +
                    gl.getShaderInfoLog(shaderObject).split("\n").join("\n    ") +
                    "\nThe expanded shader source code was:\n\n" +
                    shaderSource);
            }
            return shaderObject;
        };
        WebGL_Shader.prototype.resolveShaderSource = function (shaderDict, name) {
            if (!(name in shaderDict))
                throw new Error("Unable to find shader source for '" + name + "'");
            var shaderSource = shaderDict[name];
            /* Rudimentary include handling for convenience.
               Not the most robust, but it will do for our purposes */
            var pattern = new RegExp('#include "(.+)"');
            var match;
            while (match = pattern.exec(shaderSource)) {
                shaderSource = shaderSource.slice(0, match.index) +
                    this.resolveShaderSource(shaderDict, match[1]) +
                    shaderSource.slice(match.index + match[0].length);
            }
            return shaderSource;
        };
        WebGL_Shader.prototype.uniformIndex = function (name) {
            if (!(name in this.uniforms))
                this.uniforms[name] = gl.getUniformLocation(this.program, name);
            return this.uniforms[name];
        };
        WebGL_Shader.prototype.uniformTexture = function (name, texture) {
            var id = this.uniformIndex(name);
            if (id != -1)
                gl.uniform1i(id, texture.boundUnit);
        };
        WebGL_Shader.prototype.uniformF = function (name, f) {
            var id = this.uniformIndex(name);
            if (id != -1)
                gl.uniform1f(id, f);
        };
        WebGL_Shader.prototype.uniform2F = function (name, f1, f2) {
            var id = this.uniformIndex(name);
            if (id != -1)
                gl.uniform2f(id, f1, f2);
        };
        return WebGL_Shader;
    }());
    EcognitaMathLib.WebGL_Shader = WebGL_Shader;
    //add attribute -> init -> copy -> bind -> draw -> release
    var WebGL_VertexBuffer = /** @class */ (function () {
        function WebGL_VertexBuffer() {
            this.attributes = [];
            this.elementSize = 0;
        }
        WebGL_VertexBuffer.prototype.bind = function (shader) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.glName);
            for (var i = 0; i < this.attributes.length; ++i) {
                this.attributes[i].index = gl.getAttribLocation(shader.program, this.attributes[i].name);
                if (this.attributes[i].index >= 0) {
                    var attr = this.attributes[i];
                    gl.enableVertexAttribArray(attr.index);
                    gl.vertexAttribPointer(attr.index, attr.size, attr.type, attr.norm, this.elementSize, attr.offset);
                }
            }
        };
        WebGL_VertexBuffer.prototype.release = function () {
            for (var i = 0; i < this.attributes.length; ++i) {
                if (this.attributes[i].index >= 0) {
                    gl.disableVertexAttribArray(this.attributes[i].index);
                    this.attributes[i].index = -1;
                }
            }
        };
        WebGL_VertexBuffer.prototype.addAttribute = function (name, size, type, norm) {
            this.attributes.push({
                "name": name,
                "size": size,
                "type": type,
                "norm": norm,
                "offset": this.elementSize,
                "index": -1
            });
            this.elementSize += size * GetGLTypeSize(type);
        };
        WebGL_VertexBuffer.prototype.addAttributes = function (attrArray, sizeArray) {
            for (var i = 0; i < attrArray.length; i++) {
                this.addAttribute(attrArray[i], sizeArray[i], gl.FLOAT, false);
            }
        };
        WebGL_VertexBuffer.prototype.init = function (numVerts) {
            this.length = numVerts;
            this.glName = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.glName);
            gl.bufferData(gl.ARRAY_BUFFER, this.length * this.elementSize, gl.STATIC_DRAW);
        };
        WebGL_VertexBuffer.prototype.copy = function (data) {
            data = new Float32Array(data);
            if (data.byteLength != this.length * this.elementSize)
                throw new Error("Resizing VBO during copy strongly discouraged");
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        };
        WebGL_VertexBuffer.prototype.draw = function (mode, length) {
            gl.drawArrays(mode, 0, length ? length : this.length);
        };
        return WebGL_VertexBuffer;
    }());
    EcognitaMathLib.WebGL_VertexBuffer = WebGL_VertexBuffer;
    var WebGL_IndexBuffer = /** @class */ (function () {
        function WebGL_IndexBuffer() {
        }
        WebGL_IndexBuffer.prototype.bind = function () {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.glName);
        };
        WebGL_IndexBuffer.prototype.init = function (index) {
            this.length = index.length;
            this.glName = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.glName);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int16Array(index), gl.STATIC_DRAW);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        };
        WebGL_IndexBuffer.prototype.draw = function (mode, length) {
            gl.drawElements(mode, length ? length : this.length, gl.UNSIGNED_SHORT, 0);
        };
        return WebGL_IndexBuffer;
    }());
    EcognitaMathLib.WebGL_IndexBuffer = WebGL_IndexBuffer;
    var WebGL_FrameBuffer = /** @class */ (function () {
        function WebGL_FrameBuffer(width, height) {
            this.width = width;
            this.height = height;
            var frameBuffer = gl.createFramebuffer();
            this.framebuffer = frameBuffer;
            var depthRenderBuffer = gl.createRenderbuffer();
            this.depthbuffer = depthRenderBuffer;
            var fTexture = gl.createTexture();
            this.targetTexture = fTexture;
        }
        WebGL_FrameBuffer.prototype.bindFrameBuffer = function () {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        };
        WebGL_FrameBuffer.prototype.bindDepthBuffer = function () {
            gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthbuffer);
            //setiing render buffer to depth buffer
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.width, this.height);
            //attach depthbuffer to framebuffer
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthbuffer);
        };
        WebGL_FrameBuffer.prototype.renderToTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            //make sure we have enought memory to render the width x height size texture
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            //texture settings
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            //attach framebuff to texture
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        WebGL_FrameBuffer.prototype.renderToShadowTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            //make sure we have enought memory to render the width x height size texture
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            //texture settings
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            //attach framebuff to texture
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        WebGL_FrameBuffer.prototype.renderToFloatTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            //make sure we have enought memory to render the width x height size texture
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.FLOAT, null);
            //texture settings
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            //attach framebuff to texture
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        WebGL_FrameBuffer.prototype.renderToCubeTexture = function (cubeTarget) {
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.targetTexture);
            for (var i = 0; i < cubeTarget.length; i++) {
                gl.texImage2D(cubeTarget[i], 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        };
        WebGL_FrameBuffer.prototype.releaseCubeTex = function () {
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
            gl.bindRenderbuffer(gl.RENDERBUFFER, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        };
        WebGL_FrameBuffer.prototype.release = function () {
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.bindRenderbuffer(gl.RENDERBUFFER, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        };
        return WebGL_FrameBuffer;
    }());
    EcognitaMathLib.WebGL_FrameBuffer = WebGL_FrameBuffer;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var Shaders = {
    'Abstraction-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D akf;\n' +
        'uniform sampler2D fxdog;\n' +
        'uniform vec3 edge_color;\n\n' +
        'uniform bool b_Abstraction;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '	vec2 uv = gl_FragCoord.xy / src_size ; \n' +
        '    vec2 uv_src = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y' +
        ') / src_size.y);\n' +
        '    if(b_Abstraction){\n' +
        '        vec2 d = 1.0 / src_size;\n' +
        '        vec3 c = texture2D(akf, uv).xyz;\n' +
        '        float e = texture2D(fxdog, uv).x;\n' +
        '        gl_FragColor = vec4(mix(edge_color, c, e), 1.0);\n' +
        '    }else{\n' +
        '        gl_FragColor = texture2D(src, uv_src);\n' +
        '    }\n' +
        '}\n',
    'Abstraction-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'AKF-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D k0;\n' +
        'uniform sampler2D tfm;\n' +
        'uniform float radius;\n' +
        'uniform float q;\n' +
        'uniform float alpha;\n\n' +
        'uniform bool anisotropic;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'const float PI = 3.14159265358979323846;\n' +
        'const int N = 8;\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '	vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src' +
        '_size.y);\n\n' +
        '    if(anisotropic){\n' +
        '        vec4 m[8];\n' +
        '        vec3 s[8];\n' +
        '        for (int k = 0; k < N; ++k) {\n' +
        '            m[k] = vec4(0.0);\n' +
        '            s[k] = vec3(0.0);\n' +
        '        }\n\n' +
        '        float piN = 2.0 * PI / float(N);\n' +
        '        mat2 X = mat2(cos(piN), sin(piN), -sin(piN), cos(piN));\n\n' +
        '        vec4 t = texture2D(tfm, uv);\n' +
        '        float a = radius * clamp((alpha + t.w) / alpha, 0.1, 2.0); \n' +
        '        float b = radius * clamp(alpha / (alpha + t.w), 0.1, 2.0);\n\n' +
        '        float cos_phi = cos(t.z);\n' +
        '        float sin_phi = sin(t.z);\n\n' +
        '        mat2 R = mat2(cos_phi, -sin_phi, sin_phi, cos_phi);\n' +
        '        mat2 S = mat2(0.5/a, 0.0, 0.0, 0.5/b);\n' +
        '        mat2 SR = S * R;\n\n' +
        '        const int max_x = 6;\n' +
        '        const int max_y = 6;\n\n' +
        '        for (int j = -max_y; j <= max_y; ++j) {\n' +
        '            for (int i = -max_x; i <= max_x; ++i) {\n' +
        '                vec2 v = SR * vec2(i,j);\n' +
        '                if (dot(v,v) <= 0.25) {\n' +
        '                vec4 c_fix = texture2D(src, uv + vec2(i,j) / src_size);\n' +
        '                vec3 c = c_fix.rgb;\n' +
        '                for (int k = 0; k < N; ++k) {\n' +
        '                    float w = texture2D(k0, vec2(0.5, 0.5) + v).x;\n\n' +
        '                    m[k] += vec4(c * w, w);\n' +
        '                    s[k] += c * c * w;\n\n' +
        '                    v *= X;\n' +
        '                    }\n' +
        '                }\n' +
        '            }\n' +
        '        }\n\n' +
        '        vec4 o = vec4(0.0);\n' +
        '        for (int k = 0; k < N; ++k) {\n' +
        '            m[k].rgb /= m[k].w;\n' +
        '            s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);\n\n' +
        '            float sigma2 = s[k].r + s[k].g + s[k].b;\n' +
        '            float w = 1.0 / (1.0 + pow(255.0 * sigma2, 0.5 * q));\n\n' +
        '            o += vec4(m[k].rgb * w, w);\n' +
        '        }\n\n' +
        '        gl_FragColor = vec4(o.rgb / o.w, 1.0);\n' +
        '    }else{\n' +
        '        gl_FragColor = texture2D(src, uv);\n' +
        '    }\n\n' +
        '}\n',
    'AKF-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'Anisotropic-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n' +
        'uniform sampler2D visual;\n' +
        'uniform bool anisotropic;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main (void) {\n' +
        '	vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '	vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src' +
        '_size.y);\n' +
        '	vec4 t = texture2D( tfm, uv );\n\n' +
        '	if(anisotropic){\n' +
        '		gl_FragColor = texture2D(visual, vec2(t.w,0.5));\n' +
        '	}else{\n' +
        '		gl_FragColor = texture2D(src, uv);\n' +
        '	}\n' +
        '}\n',
    'Anisotropic-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'blurEffect-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'varying vec4      vColor;\n\n' +
        'void main(void){\n' +
        '	vec2 tFrag = vec2(1.0 / 512.0);\n' +
        '	vec4 destColor = texture2D(texture, gl_FragCoord.st * tFrag);\n' +
        '	destColor *= 0.36;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0,  1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  0.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  0.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0, -1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0, -1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0, -1.0)) * tFrag) *' +
        ' 0.04;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0,  2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  1.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  1.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  0.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  0.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0, -1.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0, -1.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0, -2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0, -2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0, -2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0, -2.0)) * tFrag) *' +
        ' 0.02;\n' +
        '	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0, -2.0)) * tFrag) *' +
        ' 0.02;\n\n' +
        '	gl_FragColor = vColor * destColor;\n' +
        '}\n',
    'blurEffect-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vColor      = color;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'bumpMapping-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'varying vec4      vColor;\n' +
        'varying vec2      vTextureCoord;\n' +
        'varying vec3      vEyeDirection;\n' +
        'varying vec3      vLightDirection;\n\n' +
        'void main(void){\n' +
        '	vec3 mNormal    = (texture2D(texture, vTextureCoord) * 2.0 - 1.0).rgb;\n' +
        '	vec3 light      = normalize(vLightDirection);\n' +
        '	vec3 eye        = normalize(vEyeDirection);\n' +
        '	vec3 halfLE     = normalize(light + eye);\n' +
        '	float diffuse   = clamp(dot(mNormal, light), 0.1, 1.0);\n' +
        '	float specular  = pow(clamp(dot(mNormal, halfLE), 0.0, 1.0), 50.0);\n' +
        '	vec4  destColor = vColor * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0)' +
        ';\n' +
        '	gl_FragColor    = destColor;\n' +
        '}\n',
    'bumpMapping-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'attribute vec2 textureCoord;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 invMatrix;\n' +
        'uniform   vec3 lightPosition;\n' +
        'uniform   vec3 eyePosition;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec2 vTextureCoord;\n' +
        'varying   vec3 vEyeDirection;\n' +
        'varying   vec3 vLightDirection;\n\n' +
        'void main(void){\n' +
        '	vec3 pos      = (mMatrix * vec4(position, 0.0)).xyz;\n' +
        '	vec3 invEye   = (invMatrix * vec4(eyePosition, 0.0)).xyz;\n' +
        '	vec3 invLight = (invMatrix * vec4(lightPosition, 0.0)).xyz;\n' +
        '	vec3 eye      = invEye - pos;\n' +
        '	vec3 light    = invLight - pos;\n' +
        '	vec3 n = normalize(normal);\n' +
        '	vec3 t = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));\n' +
        '	vec3 b = cross(n, t);\n' +
        '	vEyeDirection.x   = dot(t, eye);\n' +
        '	vEyeDirection.y   = dot(b, eye);\n' +
        '	vEyeDirection.z   = dot(n, eye);\n' +
        '	normalize(vEyeDirection);\n' +
        '	vLightDirection.x = dot(t, light);\n' +
        '	vLightDirection.y = dot(b, light);\n' +
        '	vLightDirection.z = dot(n, light);\n' +
        '	normalize(vLightDirection);\n' +
        '	vColor         = color;\n' +
        '	vTextureCoord  = textureCoord;\n' +
        '	gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'cubeTexBumpMapping-frag': 'precision mediump float;\n\n' +
        'uniform vec3        eyePosition;\n' +
        'uniform sampler2D   normalMap;\n' +
        'uniform samplerCube cubeTexture;\n' +
        'uniform bool        reflection;\n' +
        'varying vec3        vPosition;\n' +
        'varying vec2        vTextureCoord;\n' +
        'varying vec3        vNormal;\n' +
        'varying vec3        tTangent;\n\n' +
        'varying vec4        vColor;\n\n' +
        '//reflect = I - 2.0 * dot(N, I) * N.\n' +
        'vec3 egt_reflect(vec3 p, vec3 n){\n' +
        '  return  p - 2.0* dot(n,p) * n;\n' +
        '}\n\n' +
        'void main(void){\n' +
        '	vec3 tBinormal = cross(vNormal, tTangent);\n' +
        '	mat3 mView     = mat3(tTangent, tBinormal, vNormal);\n' +
        '	vec3 mNormal   = mView * (texture2D(normalMap, vTextureCoord) * 2.0 - 1.0).rgb;\n' +
        '	vec3 ref;\n' +
        '	if(reflection){\n' +
        '		ref = reflect(vPosition - eyePosition, mNormal);\n' +
        '        //ref = egt_reflect(normalize(vPosition - eyePosition),normalize(vNormal' +
        '));\n' +
        '	}else{\n' +
        '		ref = vNormal;\n' +
        '	}\n' +
        '	vec4 envColor  = textureCube(cubeTexture, ref);\n' +
        '	vec4 destColor = vColor * envColor;\n' +
        '	gl_FragColor   = destColor;\n' +
        '}\n',
    'cubeTexBumpMapping-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'attribute vec2 textureCoord;\n\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec3 vPosition;\n' +
        'varying   vec2 vTextureCoord;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec3 tTangent;\n\n' +
        'void main(void){\n' +
        '	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n' +
        '	vNormal     = (mMatrix * vec4(normal, 0.0)).xyz;\n' +
        '	vTextureCoord = textureCoord;\n' +
        '	vColor      = color;\n' +
        '	tTangent      = cross(vNormal, vec3(0.0, 1.0, 0.0));\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'cubeTexMapping-frag': 'precision mediump float;\n\n' +
        'uniform vec3        eyePosition;\n' +
        'uniform samplerCube cubeTexture;\n' +
        'uniform bool        reflection;\n' +
        'varying vec3        vPosition;\n' +
        'varying vec3        vNormal;\n' +
        'varying vec4        vColor;\n\n' +
        '//reflect = I - 2.0 * dot(N, I) * N.\n' +
        'vec3 egt_reflect(vec3 p, vec3 n){\n' +
        '  return  p - 2.0* dot(n,p) * n;\n' +
        '}\n\n' +
        'void main(void){\n' +
        '	vec3 ref;\n' +
        '	if(reflection){\n' +
        '		ref = reflect(vPosition - eyePosition, vNormal);\n' +
        '        //ref = egt_reflect(normalize(vPosition - eyePosition),normalize(vNormal' +
        '));\n' +
        '	}else{\n' +
        '		ref = vNormal;\n' +
        '	}\n' +
        '	vec4 envColor  = textureCube(cubeTexture, ref);\n' +
        '	vec4 destColor = vColor * envColor;\n' +
        '	gl_FragColor   = destColor;\n' +
        '}\n',
    'cubeTexMapping-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec3 vPosition;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n' +
        '	vNormal     = (mMatrix * vec4(normal, 0.0)).xyz;\n' +
        '	vColor      = color;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'demo-frag': 'void main(void){\n' +
        '	gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n' +
        '}\n',
    'demo-vert': 'attribute vec3 position;\n' +
        'uniform   mat4 mvpMatrix;\n\n' +
        'void main(void){\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'demo1-frag': 'precision mediump float;\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'demo1-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vColor = color;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'directionLighting-frag': 'precision mediump float;\n\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'directionLighting-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec3 normal;\n\n' +
        'uniform mat4 mvpMatrix;\n' +
        'uniform mat4 invMatrix;\n' +
        'uniform vec3 lightDirection;\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0)).xyz;\n' +
        '    float diffuse = clamp(dot(invLight,normal),0.1,1.0);\n' +
        '    vColor = color*vec4(vec3(diffuse),1.0);\n' +
        '    gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'dir_ambient-frag': 'precision mediump float;\n\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'dir_ambient-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec3 normal;\n\n' +
        'uniform mat4 mvpMatrix;\n' +
        'uniform mat4 invMatrix;\n' +
        'uniform vec3 lightDirection;\n' +
        'uniform vec4 ambientColor;\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0)).xyz;\n' +
        '    float diffuse = clamp(dot(invLight,normal),0.1,1.0);\n' +
        '    vColor = color*vec4(vec3(diffuse),1.0) +ambientColor;\n' +
        '    gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'DoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n\n' +
        'uniform bool b_DoG;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma_e;\n' +
        'uniform float sigma_r;\n' +
        'uniform float tau;\n' +
        'uniform float phi;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    if(b_DoG){\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2 uv = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n' +
        '        float twoSigmaESquared = 2.0 * sigma_e * sigma_e;\n' +
        '        float twoSigmaRSquared = 2.0 * sigma_r * sigma_r;\n' +
        '        int halfWidth = int(ceil( 2.0 * sigma_r ));\n\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        vec2 sum = vec2(0.0);\n' +
        '        vec2 norm = vec2(0.0);\n\n' +
        '        for(int cnt=0;cnt<MAX_NUM_ITERATION;cnt++){\n' +
        '            if(cnt > (2*halfWidth+1)*(2*halfWidth+1)){break;}\n' +
        '            int i = int(cnt / (2*halfWidth+1)) - halfWidth;\n' +
        '            int j = cnt - halfWidth - int(cnt / (2*halfWidth+1)) * (2*halfWidth+' +
        '1);\n\n' +
        '            float d = length(vec2(i,j));\n' +
        '            vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n' +
        '                                exp( -d * d / twoSigmaRSquared ));\n\n' +
        '            vec2 L = texture2D(src, (uv + vec2(i,j)) * Frag).xx;\n\n' +
        '            norm += 2.0 * kernel;\n' +
        '            sum += kernel * L;\n' +
        '        }\n\n' +
        '        sum /= norm;\n\n' +
        '        float H = 100.0 * (sum.x - tau * sum.y);\n' +
        '        float edge = ( H > 0.0 )? 1.0 : 2.0 * smoothstep(-2.0, 2.0, phi * H );\n' +
        '        destColor = vec3(edge);\n' +
        '    }else{\n' +
        '        destColor = texture2D(src, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'DoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'ETF-frag': '// Edge Tangent Flow\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 uv = gl_FragCoord.xy / src_size;\n' +
        '    vec2 d = 1.0 / src_size;\n' +
        '    vec3 c = texture2D(src, uv).xyz;\n' +
        '    float gx = c.z;\n' +
        '    vec2 tx = c.xy;\n' +
        '    const float KERNEL = 5.0;\n' +
        '    vec2 etf = vec2(0.0);\n' +
        '    vec2 sum = vec2(0,0);\n' +
        '    float weight = 0.0;\n\n' +
        '    for(float j = -KERNEL ; j<KERNEL;j++){\n' +
        '        for(float i=-KERNEL ; i<KERNEL;i++){\n' +
        '            vec2 ty = texture2D(src, uv + vec2(i * d.x, j * d.y)).xy;\n' +
        '            float gy = texture2D(src, uv + vec2(i * d.x, j * d.y)).z;\n\n' +
        '            float wd = abs(dot(tx,ty));\n' +
        '            float wm = (gy - gx + 1.0)/2.0;\n' +
        '            float phi = dot(gx,gy)>0.0?1.0:-1.0;\n' +
        '            float ws = sqrt(j*j+i*i) < KERNEL?1.0:0.0;\n\n' +
        '            sum += ty * (wm * wd );\n' +
        '            weight += wm * wd ;\n' +
        '        }\n' +
        '    }\n\n' +
        '    if(weight != 0.0){\n' +
        '        etf = sum / weight;\n' +
        '    }else{\n' +
        '        etf = vec2(0.0);\n' +
        '    }\n\n' +
        '    float mag = sqrt(etf.x*etf.x + etf.y*etf.y);\n' +
        '    float vx = etf.x/mag;\n' +
        '    float vy = etf.y/mag;\n' +
        '    gl_FragColor = vec4(vx,vy,mag, 1.0);\n' +
        '}\n',
    'ETF-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'FDoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma_m;\n' +
        'uniform float phi;\n\n' +
        'uniform bool b_FDoG;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'struct lic_t { \n' +
        '    vec2 p; \n' +
        '    vec2 t;\n' +
        '    float w;\n' +
        '    float dw;\n' +
        '};\n\n' +
        'void step(inout lic_t s) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 t = texture2D(tfm, s.p).xy;\n' +
        '    if (dot(t, s.t) < 0.0) t = -t;\n' +
        '    s.t = t;\n\n' +
        '    s.dw = (abs(t.x) > abs(t.y))? \n' +
        '        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n' +
        '        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n' +
        '    s.p += t * s.dw / src_size;\n' +
        '    s.w += s.dw;\n' +
        '}\n\n' +
        'void main (void) {\n\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    if(b_FDoG){\n' +
        '        vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '        vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y' +
        ') / src_size.y);\n\n' +
        '        float twoSigmaMSquared = 2.0 * sigma_m * sigma_m;\n' +
        '        float halfWidth = 2.0 * sigma_m;\n\n' +
        '        float H = texture2D( src, uv ).x;\n' +
        '        float w = 1.0;\n\n' +
        '        lic_t a, b;\n' +
        '        a.p = b.p = uv;\n' +
        '        a.t = texture2D( tfm, uv ).xy / src_size;\n' +
        '        b.t = -a.t;\n' +
        '        a.w = b.w = 0.0;\n\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (a.w < halfWidth) {\n' +
        '                step(a);\n' +
        '                float k = a.dw * exp(-a.w * a.w / twoSigmaMSquared);\n' +
        '                H += k * texture2D(src, a.p).x;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (b.w < halfWidth) {\n' +
        '                step(b);\n' +
        '                float k = b.dw * exp(-b.w * b.w / twoSigmaMSquared);\n' +
        '                H += k * texture2D(src, b.p).x;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n' +
        '        H /= w;\n' +
        '        float edge = ( H > 0.0 )? 1.0 : 2.0 * smoothstep(-2.0, 2.0, phi * H );\n' +
        '        destColor = vec3(edge);\n' +
        '    }\n' +
        '    else{\n' +
        '        destColor = texture2D(src, vTexCoord).rgb;\n' +
        '    }\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'FDoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'filterScene-frag': 'precision mediump float;\n\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'filterScene-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 invMatrix;\n' +
        'uniform   vec3 lightDirection;\n' +
        'uniform   vec3 eyeDirection;\n' +
        'uniform   vec4 ambientColor;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n' +
        '	vec3  invEye   = normalize(invMatrix * vec4(eyeDirection, 0.0)).xyz;\n' +
        '	vec3  halfLE   = normalize(invLight + invEye);\n' +
        '	float diffuse  = clamp(dot(normal, invLight), 0.0, 1.0);\n' +
        '	float specular = pow(clamp(dot(normal, halfLE), 0.0, 1.0), 50.0);\n' +
        '	vec4  amb      = color * ambientColor;\n' +
        '	vColor         = amb * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0);\n' +
        '	gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'frameBuffer-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'varying vec4      vColor;\n' +
        'varying vec2      vTextureCoord;\n\n' +
        'void main(void){\n' +
        '	vec4 smpColor = texture2D(texture, vTextureCoord);\n' +
        '	gl_FragColor  = vColor * smpColor;\n' +
        '}\n',
    'frameBuffer-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'attribute vec2 textureCoord;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 invMatrix;\n' +
        'uniform   vec3 lightDirection;\n' +
        'uniform   bool useLight;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec2 vTextureCoord;\n\n' +
        'void main(void){\n' +
        '	if(useLight){\n' +
        '		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n' +
        '		float diffuse  = clamp(dot(normal, invLight), 0.2, 1.0);\n' +
        '		vColor         = vec4(color.xyz * vec3(diffuse), 1.0);\n' +
        '	}else{\n' +
        '		vColor         = color;\n' +
        '	}\n' +
        '	vTextureCoord  = textureCoord;\n' +
        '	gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'FXDoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma_m;\n' +
        'uniform float phi;\n' +
        'uniform float epsilon;\n\n' +
        'uniform bool b_FXDoG;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'float cosh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float cosH = (tmp + 1.0 / tmp) / 2.0;\n' +
        '    return cosH;\n' +
        '}\n\n' +
        'float tanh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n' +
        '    return tanH;\n' +
        '}\n\n' +
        'float sinh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float sinH = (tmp - 1.0 / tmp) / 2.0;\n' +
        '    return sinH;\n' +
        '}\n\n' +
        'struct lic_t { \n' +
        '    vec2 p; \n' +
        '    vec2 t;\n' +
        '    float w;\n' +
        '    float dw;\n' +
        '};\n\n' +
        'void step(inout lic_t s) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 t = texture2D(tfm, s.p).xy;\n' +
        '    if (dot(t, s.t) < 0.0) t = -t;\n' +
        '    s.t = t;\n\n' +
        '    s.dw = (abs(t.x) > abs(t.y))? \n' +
        '        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n' +
        '        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n' +
        '    s.p += t * s.dw / src_size;\n' +
        '    s.w += s.dw;\n' +
        '}\n\n' +
        'void main (void) {\n\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    if(b_FXDoG){\n' +
        '        vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '        vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y' +
        ') / src_size.y);\n\n' +
        '        float twoSigmaMSquared = 2.0 * sigma_m * sigma_m;\n' +
        '        float halfWidth = 2.0 * sigma_m;\n\n' +
        '        float H = texture2D( src, uv ).x;\n' +
        '        float w = 1.0;\n\n' +
        '        lic_t a, b;\n' +
        '        a.p = b.p = uv;\n' +
        '        a.t = texture2D( tfm, uv ).xy / src_size;\n' +
        '        b.t = -a.t;\n' +
        '        a.w = b.w = 0.0;\n\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (a.w < halfWidth) {\n' +
        '                step(a);\n' +
        '                float k = a.dw * exp(-a.w * a.w / twoSigmaMSquared);\n' +
        '                H += k * texture2D(src, a.p).x;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (b.w < halfWidth) {\n' +
        '                step(b);\n' +
        '                float k = b.dw * exp(-b.w * b.w / twoSigmaMSquared);\n' +
        '                H += k * texture2D(src, b.p).x;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n' +
        '        H /= w;\n' +
        '        float edge = ( H > epsilon )? 1.0 : 1.0 + tanh( phi * (H - epsilon));\n' +
        '        destColor = vec3(edge);\n' +
        '    }\n' +
        '    else{\n' +
        '        destColor = texture2D(src, vTexCoord).rgb;\n' +
        '    }\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'FXDoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'gaussianFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'uniform bool b_gaussian;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'uniform float weight[10];\n' +
        'uniform bool horizontal;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '    vec3  destColor = vec3(0.0);\n' +
        '	if(b_gaussian){\n' +
        '		float tFrag = 1.0 / cvsHeight;\n' +
        '		float sFrag = 1.0 / cvsWidth;\n' +
        '		vec2  Frag = vec2(sFrag,tFrag);\n' +
        '		vec2 fc;\n' +
        '		if(horizontal){\n' +
        '			fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n' +
        '			destColor += texture2D(texture, (fc + vec2(-9.0, 0.0)) * Frag).rgb * weight[9' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-8.0, 0.0)) * Frag).rgb * weight[8' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-7.0, 0.0)) * Frag).rgb * weight[7' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-6.0, 0.0)) * Frag).rgb * weight[6' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-5.0, 0.0)) * Frag).rgb * weight[5' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-4.0, 0.0)) * Frag).rgb * weight[4' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-3.0, 0.0)) * Frag).rgb * weight[3' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-2.0, 0.0)) * Frag).rgb * weight[2' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(-1.0, 0.0)) * Frag).rgb * weight[1' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 0.0, 0.0)) * Frag).rgb * weight[0' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 1.0, 0.0)) * Frag).rgb * weight[1' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 2.0, 0.0)) * Frag).rgb * weight[2' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 3.0, 0.0)) * Frag).rgb * weight[3' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 4.0, 0.0)) * Frag).rgb * weight[4' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 5.0, 0.0)) * Frag).rgb * weight[5' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 6.0, 0.0)) * Frag).rgb * weight[6' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 7.0, 0.0)) * Frag).rgb * weight[7' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 8.0, 0.0)) * Frag).rgb * weight[8' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2( 9.0, 0.0)) * Frag).rgb * weight[9' +
        '];\n' +
        '		}else{\n' +
        '			fc = gl_FragCoord.st;\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -9.0)) * Frag).rgb * weight[9' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -8.0)) * Frag).rgb * weight[8' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -7.0)) * Frag).rgb * weight[7' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -6.0)) * Frag).rgb * weight[6' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -5.0)) * Frag).rgb * weight[5' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -4.0)) * Frag).rgb * weight[4' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -3.0)) * Frag).rgb * weight[3' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -2.0)) * Frag).rgb * weight[2' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0, -1.0)) * Frag).rgb * weight[1' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  0.0)) * Frag).rgb * weight[0' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  1.0)) * Frag).rgb * weight[1' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  2.0)) * Frag).rgb * weight[2' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  3.0)) * Frag).rgb * weight[3' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  4.0)) * Frag).rgb * weight[4' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  5.0)) * Frag).rgb * weight[5' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  6.0)) * Frag).rgb * weight[6' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  7.0)) * Frag).rgb * weight[7' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  8.0)) * Frag).rgb * weight[8' +
        '];\n' +
        '			destColor += texture2D(texture, (fc + vec2(0.0,  9.0)) * Frag).rgb * weight[9' +
        '];\n' +
        '		}\n' +
        '	}else{\n' +
        ' 		destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '	}\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'gaussianFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'Gaussian_K-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform float sigma;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 uv = gl_FragCoord.xy / src_size;\n\n' +
        '    float twoSigma2 = 2.0 * 2.0 * 2.0;\n' +
        '    const int halfWidth = 4;//int(ceil( 2.0 * sigma ));\n\n' +
        '    vec3 sum = vec3(0.0);\n' +
        '    float norm = 0.0;\n' +
        '    for ( int i = -halfWidth; i <= halfWidth; ++i ) {\n' +
        '        for ( int j = -halfWidth; j <= halfWidth; ++j ) {\n' +
        '            float d = length(vec2(i,j));\n' +
        '            float kernel = exp( -d *d / twoSigma2 );\n' +
        '            vec3 c = texture2D(src, uv + vec2(i,j) / src_size ).rgb;\n' +
        '            sum += kernel * c;\n' +
        '            norm += kernel;\n' +
        '        }\n' +
        '    }\n' +
        '    gl_FragColor = vec4(sum / norm, 1.0);\n' +
        '}\n',
    'Gaussian_K-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'gkuwaharaFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n\n' +
        'uniform float weight[49];\n' +
        'uniform bool b_gkuwahara;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '    vec3  destColor = vec3(0.0);\n' +
        '    if(b_gkuwahara){\n' +
        '        float q = 3.0;\n' +
        '        vec3 mean[8];\n' +
        '        vec3 sigma[8];\n' +
        '        vec2 offset[49];\n' +
        '        offset[0] = vec2(-3.0, -3.0);\n' +
        '        offset[1] = vec2(-2.0, -3.0);\n' +
        '        offset[2] = vec2(-1.0, -3.0);\n' +
        '        offset[3] = vec2( 0.0, -3.0);\n' +
        '        offset[4] = vec2( 1.0, -3.0);\n' +
        '        offset[5] = vec2( 2.0, -3.0);\n' +
        '        offset[6] = vec2( 3.0, -3.0);\n\n' +
        '        offset[7]  = vec2(-3.0, -2.0);\n' +
        '        offset[8]  = vec2(-2.0, -2.0);\n' +
        '        offset[9]  = vec2(-1.0, -2.0);\n' +
        '        offset[10] = vec2( 0.0, -2.0);\n' +
        '        offset[11] = vec2( 1.0, -2.0);\n' +
        '        offset[12] = vec2( 2.0, -2.0);\n' +
        '        offset[13] = vec2( 3.0, -2.0);\n\n' +
        '        offset[14] = vec2(-3.0, -1.0);\n' +
        '        offset[15] = vec2(-2.0, -1.0);\n' +
        '        offset[16] = vec2(-1.0, -1.0);\n' +
        '        offset[17] = vec2( 0.0, -1.0);\n' +
        '        offset[18] = vec2( 1.0, -1.0);\n' +
        '        offset[19] = vec2( 2.0, -1.0);\n' +
        '        offset[20] = vec2( 3.0, -1.0);\n\n' +
        '        offset[21] = vec2(-3.0,  0.0);\n' +
        '        offset[22] = vec2(-2.0,  0.0);\n' +
        '        offset[23] = vec2(-1.0,  0.0);\n' +
        '        offset[24] = vec2( 0.0,  0.0);\n' +
        '        offset[25] = vec2( 1.0,  0.0);\n' +
        '        offset[26] = vec2( 2.0,  0.0);\n' +
        '        offset[27] = vec2( 3.0,  0.0);\n\n' +
        '        offset[28] = vec2(-3.0,  1.0);\n' +
        '        offset[29] = vec2(-2.0,  1.0);\n' +
        '        offset[30] = vec2(-1.0,  1.0);\n' +
        '        offset[31] = vec2( 0.0,  1.0);\n' +
        '        offset[32] = vec2( 1.0,  1.0);\n' +
        '        offset[33] = vec2( 2.0,  1.0);\n' +
        '        offset[34] = vec2( 3.0,  1.0);\n\n' +
        '        offset[35] = vec2(-3.0,  2.0);\n' +
        '        offset[36] = vec2(-2.0,  2.0);\n' +
        '        offset[37] = vec2(-1.0,  2.0);\n' +
        '        offset[38] = vec2( 0.0,  2.0);\n' +
        '        offset[39] = vec2( 1.0,  2.0);\n' +
        '        offset[40] = vec2( 2.0,  2.0);\n' +
        '        offset[41] = vec2( 3.0,  2.0);\n\n' +
        '        offset[42] = vec2(-3.0,  3.0);\n' +
        '        offset[43] = vec2(-2.0,  3.0);\n' +
        '        offset[44] = vec2(-1.0,  3.0);\n' +
        '        offset[45] = vec2( 0.0,  3.0);\n' +
        '        offset[46] = vec2( 1.0,  3.0);\n' +
        '        offset[47] = vec2( 2.0,  3.0);\n' +
        '        offset[48] = vec2( 3.0,  3.0);\n\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n' +
        '        vec3 cur_std = vec3(0.0);\n' +
        '        float cur_weight = 0.0;\n' +
        '        vec3 total_ms = vec3(0.0);\n' +
        '        vec3 total_s = vec3(0.0);\n\n' +
        '        mean[0]=vec3(0.0);\n' +
        '        sigma[0]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * weight[24' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[24]) * Frag).rgb * weight[24];\n' +
        '        cur_weight+= weight[24];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * weight[31' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[31]) * Frag).rgb * weight[31];\n' +
        '        cur_weight+= weight[31];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * weight[38' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[38]) * Frag).rgb * weight[38];\n' +
        '        cur_weight+= weight[38];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * weight[45' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[45]) * Frag).rgb * weight[45];\n' +
        '        cur_weight+= weight[45];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * weight[39' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[39]) * Frag).rgb * weight[39];\n' +
        '        cur_weight+= weight[39];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * weight[46' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[46]) * Frag).rgb * weight[46];\n' +
        '        cur_weight+= weight[46];\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * weight[47' +
        '];\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[47]) * Frag).rgb * weight[47];\n' +
        '        cur_weight+= weight[47];\n\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[0] /= cur_weight;\n' +
        '            sigma[0] /= cur_weight;\n' +
        '        }\n\n' +
        '        cur_std = sigma[0] - mean[0] * mean[0];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[0] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[1]=vec3(0.0);\n' +
        '        sigma[1]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * weight[32' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[32]) * Frag).rgb * weight[32];\n' +
        '        cur_weight+= weight[32];\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * weight[33' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[33]) * Frag).rgb * weight[33];\n' +
        '        cur_weight+= weight[33];\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * weight[40' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[40]) * Frag).rgb * weight[40];\n' +
        '        cur_weight+= weight[40];\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * weight[34' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[34]) * Frag).rgb * weight[34];\n' +
        '        cur_weight+= weight[34];\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * weight[41' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[41]) * Frag).rgb * weight[41];\n' +
        '        cur_weight+= weight[41];\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * weight[48' +
        '];\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[48]) * Frag).rgb * weight[48];\n' +
        '        cur_weight+= weight[48];\n\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[1] /= cur_weight;\n' +
        '            sigma[1] /= cur_weight;\n' +
        '        }\n\n' +
        '        cur_std = sigma[1] - mean[1] * mean[1];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[1] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[2]=vec3(0.0);\n' +
        '        sigma[2]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * weight[25' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[25]) * Frag).rgb * weight[25];\n' +
        '        cur_weight+= weight[25];\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * weight[19' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[19]) * Frag).rgb * weight[19];\n' +
        '        cur_weight+= weight[19];\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * weight[26' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[26]) * Frag).rgb * weight[26];\n' +
        '        cur_weight+= weight[26];\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * weight[13' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[13]) * Frag).rgb * weight[13];\n' +
        '        cur_weight+= weight[13];\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * weight[20' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[20]) * Frag).rgb * weight[20];\n' +
        '        cur_weight+= weight[20];\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * weight[27' +
        '];\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[27]) * Frag).rgb * weight[27];\n' +
        '        cur_weight+= weight[27];\n\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[2] /= cur_weight;\n' +
        '            sigma[2] /= cur_weight;\n' +
        '        }\n\n' +
        '        cur_std = sigma[2] - mean[2] * mean[2];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[2] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[3]=vec3(0.0);\n' +
        '        sigma[3]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * weight[4];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[4]) * Frag).rgb * weight[4];\n' +
        '        cur_weight+= weight[4];\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * weight[11' +
        '];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[11]) * Frag).rgb * weight[11];\n' +
        '        cur_weight+= weight[11];\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * weight[18' +
        '];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[18]) * Frag).rgb * weight[18];\n' +
        '        cur_weight+= weight[18];\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * weight[5];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[5]) * Frag).rgb * weight[5];\n' +
        '        cur_weight+= weight[5];\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * weight[12' +
        '];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[12]) * Frag).rgb * weight[12];\n' +
        '        cur_weight+= weight[12];\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * weight[6];\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[6]) * Frag).rgb * weight[6];\n' +
        '        cur_weight+= weight[6];\n\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[3] /= cur_weight;\n' +
        '            sigma[3] /= cur_weight;\n' +
        '        }\n\n' +
        '        cur_std = sigma[3] - mean[3] * mean[3];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[3] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[4]=vec3(0.0);\n' +
        '        sigma[4]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * weight[1];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[1]) * Frag).rgb * weight[1];\n' +
        '        cur_weight+= weight[1];\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * weight[2];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[2]) * Frag).rgb * weight[2];\n' +
        '        cur_weight+= weight[2];\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * weight[9];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[9]) * Frag).rgb * weight[9];\n' +
        '        cur_weight+= weight[9];\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * weight[3];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[3]) * Frag).rgb * weight[3];\n' +
        '        cur_weight+= weight[3];\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * weight[10' +
        '];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[10]) * Frag).rgb * weight[10];\n' +
        '        cur_weight+= weight[10];\n' +
        '        mean[4]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * weight[17' +
        '];\n' +
        '        sigma[4]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[17]) * Frag).rgb * weight[17];\n' +
        '        cur_weight+= weight[17];\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[4] /= cur_weight;\n' +
        '            sigma[4] /= cur_weight;\n' +
        '        }\n' +
        '        cur_std = sigma[4] - mean[4] * mean[4];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[4] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[5]=vec3(0.0);\n' +
        '        sigma[5]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * weight[0];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[0]) * Frag).rgb * weight[0];\n' +
        '        cur_weight+= weight[0];\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * weight[7];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[7]) * Frag).rgb * weight[7];\n' +
        '        cur_weight+= weight[7];\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * weight[14' +
        '];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[14]) * Frag).rgb * weight[14];\n' +
        '        cur_weight+= weight[14];\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * weight[8];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[8]) * Frag).rgb * weight[8];\n' +
        '        cur_weight+= weight[8];\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * weight[15' +
        '];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[15]) * Frag).rgb * weight[15];\n' +
        '        cur_weight+= weight[15];\n' +
        '        mean[5]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * weight[16' +
        '];\n' +
        '        sigma[5]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[16]) * Frag).rgb * weight[16];\n' +
        '        cur_weight+= weight[16];\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[5] /= cur_weight;\n' +
        '            sigma[5] /= cur_weight;\n' +
        '        }\n' +
        '        cur_std = sigma[5] - mean[5] * mean[5];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[5] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[6]=vec3(0.0);\n' +
        '        sigma[6]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * weight[21' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[21]) * Frag).rgb * weight[21];\n' +
        '        cur_weight+= weight[21];\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * weight[28' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[28]) * Frag).rgb * weight[28];\n' +
        '        cur_weight+= weight[28];\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * weight[35' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[35]) * Frag).rgb * weight[35];\n' +
        '        cur_weight+= weight[35];\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * weight[22' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[22]) * Frag).rgb * weight[22];\n' +
        '        cur_weight+= weight[22];\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * weight[29' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[29]) * Frag).rgb * weight[29];\n' +
        '        cur_weight+= weight[29];\n' +
        '        mean[6]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * weight[23' +
        '];\n' +
        '        sigma[6]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[23]) * Frag).rgb * weight[23];\n' +
        '        cur_weight+= weight[23];\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[6] /= cur_weight;\n' +
        '            sigma[6] /= cur_weight;\n' +
        '        }\n' +
        '        cur_std = sigma[6] - mean[6] * mean[6];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n' +
        '        total_ms += mean[6] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n' +
        '        mean[7]=vec3(0.0);\n' +
        '        sigma[7]=vec3(0.0);\n' +
        '        cur_weight = 0.0;\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * weight[42' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[42]) * Frag).rgb * weight[42];\n' +
        '        cur_weight+= weight[42];\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * weight[36' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[36]) * Frag).rgb * weight[36];\n' +
        '        cur_weight+= weight[36];\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * weight[43' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[43]) * Frag).rgb * weight[43];\n' +
        '        cur_weight+= weight[43];\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * weight[30' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[30]) * Frag).rgb * weight[30];\n' +
        '        cur_weight+= weight[30];\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * weight[37' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[37]) * Frag).rgb * weight[37];\n' +
        '        cur_weight+= weight[37];\n' +
        '        mean[7]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * weight[44' +
        '];\n' +
        '        sigma[7]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[44]) * Frag).rgb * weight[44];\n' +
        '        cur_weight+= weight[44];\n' +
        '        if(cur_weight!=0.0){\n' +
        '            mean[7] /= cur_weight;\n' +
        '            sigma[7] /= cur_weight;\n' +
        '        }\n' +
        '        cur_std = sigma[7] - mean[7] * mean[7];\n' +
        '        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n' +
        '            cur_std = sqrt(cur_std);\n' +
        '        }else{\n' +
        '            cur_std = vec3(1e-10);\n' +
        '        }\n\n' +
        '        total_ms += mean[7] * pow(cur_std,vec3(-q));\n' +
        '        total_s  += pow(cur_std,vec3(-q));\n\n' +
        '        if(total_s.r> 1e-10 && total_s.g> 1e-10 && total_s.b> 1e-10){\n' +
        '            destColor = (total_ms/total_s).rgb;\n' +
        '            destColor = max(destColor, 0.0);\n' +
        '            destColor = min(destColor, 1.0);\n' +
        '        }else{\n' +
        '            destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '        }\n\n' +
        '    }else{\n' +
        '        destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'gkuwaharaFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'grayScaleFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'uniform bool      grayScale;\n' +
        'varying vec2      vTexCoord;\n\n' +
        'const float redScale   = 0.298912;\n' +
        'const float greenScale = 0.586611;\n' +
        'const float blueScale  = 0.114478;\n' +
        'const vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\n' +
        'void main(void){\n' +
        '	vec4 smpColor = texture2D(texture, vTexCoord);\n' +
        '	if(grayScale){\n' +
        '		float grayColor = dot(smpColor.rgb, monochromeScale);\n' +
        '		smpColor = vec4(vec3(grayColor), 1.0);\n' +
        '	}\n' +
        '	gl_FragColor = smpColor;\n' +
        '}\n',
    'grayScaleFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'kuwaharaFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n\n' +
        'uniform bool b_kuwahara;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '    vec3  destColor = vec3(0.0);\n' +
        '    if(b_kuwahara){\n' +
        '        float minVal =0.0;\n' +
        '        vec3 mean[4];\n' +
        '        vec3 sigma[4];\n' +
        '        vec2 offset[49];\n' +
        '        offset[0] = vec2(-3.0, -3.0);\n' +
        '        offset[1] = vec2(-2.0, -3.0);\n' +
        '        offset[2] = vec2(-1.0, -3.0);\n' +
        '        offset[3] = vec2( 0.0, -3.0);\n' +
        '        offset[4] = vec2( 1.0, -3.0);\n' +
        '        offset[5] = vec2( 2.0, -3.0);\n' +
        '        offset[6] = vec2( 3.0, -3.0);\n\n' +
        '        offset[7]  = vec2(-3.0, -2.0);\n' +
        '        offset[8]  = vec2(-2.0, -2.0);\n' +
        '        offset[9]  = vec2(-1.0, -2.0);\n' +
        '        offset[10] = vec2( 0.0, -2.0);\n' +
        '        offset[11] = vec2( 1.0, -2.0);\n' +
        '        offset[12] = vec2( 2.0, -2.0);\n' +
        '        offset[13] = vec2( 3.0, -2.0);\n\n' +
        '        offset[14] = vec2(-3.0, -1.0);\n' +
        '        offset[15] = vec2(-2.0, -1.0);\n' +
        '        offset[16] = vec2(-1.0, -1.0);\n' +
        '        offset[17] = vec2( 0.0, -1.0);\n' +
        '        offset[18] = vec2( 1.0, -1.0);\n' +
        '        offset[19] = vec2( 2.0, -1.0);\n' +
        '        offset[20] = vec2( 3.0, -1.0);\n\n' +
        '        offset[21] = vec2(-3.0,  0.0);\n' +
        '        offset[22] = vec2(-2.0,  0.0);\n' +
        '        offset[23] = vec2(-1.0,  0.0);\n' +
        '        offset[24] = vec2( 0.0,  0.0);\n' +
        '        offset[25] = vec2( 1.0,  0.0);\n' +
        '        offset[26] = vec2( 2.0,  0.0);\n' +
        '        offset[27] = vec2( 3.0,  0.0);\n\n' +
        '        offset[28] = vec2(-3.0,  1.0);\n' +
        '        offset[29] = vec2(-2.0,  1.0);\n' +
        '        offset[30] = vec2(-1.0,  1.0);\n' +
        '        offset[31] = vec2( 0.0,  1.0);\n' +
        '        offset[32] = vec2( 1.0,  1.0);\n' +
        '        offset[33] = vec2( 2.0,  1.0);\n' +
        '        offset[34] = vec2( 3.0,  1.0);\n\n' +
        '        offset[35] = vec2(-3.0,  2.0);\n' +
        '        offset[36] = vec2(-2.0,  2.0);\n' +
        '        offset[37] = vec2(-1.0,  2.0);\n' +
        '        offset[38] = vec2( 0.0,  2.0);\n' +
        '        offset[39] = vec2( 1.0,  2.0);\n' +
        '        offset[40] = vec2( 2.0,  2.0);\n' +
        '        offset[41] = vec2( 3.0,  2.0);\n\n' +
        '        offset[42] = vec2(-3.0,  3.0);\n' +
        '        offset[43] = vec2(-2.0,  3.0);\n' +
        '        offset[44] = vec2(-1.0,  3.0);\n' +
        '        offset[45] = vec2( 0.0,  3.0);\n' +
        '        offset[46] = vec2( 1.0,  3.0);\n' +
        '        offset[47] = vec2( 2.0,  3.0);\n' +
        '        offset[48] = vec2( 3.0,  3.0);\n\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n\n' +
        '        //calculate mean\n' +
        '        mean[0] = vec3(0.0);\n' +
        '        sigma[0] = vec3(0.0);\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[3]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[4]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[5]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[6]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[10]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[11]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[12]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[13]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[17]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[18]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[19]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[20]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[25]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[26]) * Frag).rgb;\n' +
        '        mean[0]  += texture2D(texture, (fc + offset[27]) * Frag).rgb;\n\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[3]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[4]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[5]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[6]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[10]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[11]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[12]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[13]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[17]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[18]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[19]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[20]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[25]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[26]) * Frag).rgb;\n' +
        '        sigma[0]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[27]) * Frag).rgb;\n\n' +
        '        mean[0] /= 16.0;\n' +
        '        sigma[0] = abs(sigma[0]/16.0 -  mean[0]* mean[0]);\n' +
        '        minVal = sigma[0].r + sigma[0].g + sigma[0].b;\n\n' +
        '        mean[1] = vec3(0.0);\n' +
        '        sigma[1] = vec3(0.0);\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[0]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[1]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[2]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[3]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[7]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[8]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[9]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[10]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[14]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[15]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[16]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[17]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[21]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[22]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[23]) * Frag).rgb;\n' +
        '        mean[1]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[0]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[1]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[2]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[3]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[7]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[8]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * texture2D' +
        '(texture, (fc + offset[9]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[10]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[14]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[15]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[16]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[17]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[21]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[22]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[23]) * Frag).rgb;\n' +
        '        sigma[1]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[24]) * Frag).rgb;\n\n' +
        '        mean[1] /= 16.0;\n' +
        '        sigma[1] = abs(sigma[1]/16.0 -  mean[1]* mean[1]);\n' +
        '        float sigmaVal = sigma[1].r + sigma[1].g + sigma[1].b;\n' +
        '        if(sigmaVal<minVal){\n' +
        '            destColor = mean[1].rgb;\n' +
        '            minVal = sigmaVal;\n' +
        '        }else{\n' +
        '            destColor = mean[0].rgb;\n' +
        '        }\n\n' +
        '        mean[2] = vec3(0.0);\n' +
        '        sigma[2] = vec3(0.0);\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[21]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[22]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[23]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[28]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[29]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[30]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[31]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[35]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[36]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[37]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[38]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[42]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[43]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[44]) * Frag).rgb;\n' +
        '        mean[2]  += texture2D(texture, (fc + offset[45]) * Frag).rgb;\n\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[21]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[22]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[23]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[28]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[29]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[30]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[31]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[35]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[36]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[37]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[38]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[42]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[43]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[44]) * Frag).rgb;\n' +
        '        sigma[2]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[45]) * Frag).rgb;\n\n' +
        '        mean[2] /= 16.0;\n' +
        '        sigma[2] = abs(sigma[2]/16.0 -  mean[2]* mean[2]);\n' +
        '        sigmaVal = sigma[2].r + sigma[2].g + sigma[2].b;\n' +
        '        if(sigmaVal<minVal){\n' +
        '            destColor = mean[2].rgb;\n' +
        '            minVal = sigmaVal;\n' +
        '        }\n\n' +
        '        mean[3] = vec3(0.0);\n' +
        '        sigma[3] = vec3(0.0);\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[25]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[26]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[27]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[31]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[32]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[33]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[34]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[38]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[39]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[40]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[41]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[45]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[46]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[47]) * Frag).rgb;\n' +
        '        mean[3]  += texture2D(texture, (fc + offset[48]) * Frag).rgb;\n\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[24]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[25]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[26]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[27]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[31]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[32]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[33]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[34]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[38]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[39]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[40]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[41]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[45]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[46]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[47]) * Frag).rgb;\n' +
        '        sigma[3]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * texture2' +
        'D(texture, (fc + offset[48]) * Frag).rgb;\n\n' +
        '        mean[3] /= 16.0;\n' +
        '        sigma[3] = abs(sigma[3]/16.0 -  mean[3]* mean[3]);\n' +
        '        sigmaVal = sigma[3].r + sigma[3].g + sigma[3].b;\n' +
        '        if(sigmaVal<minVal){\n' +
        '            destColor = mean[3].rgb;\n' +
        '            minVal = sigmaVal;\n' +
        '        }  \n\n' +
        '    }else{\n' +
        '        destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'kuwaharaFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'laplacianFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n\n' +
        'uniform bool b_laplacian;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'uniform float coef[9];\n' +
        'varying vec2 vTexCoord;\n\n' +
        'const float redScale   = 0.298912;\n' +
        'const float greenScale = 0.586611;\n' +
        'const float blueScale  = 0.114478;\n' +
        'const vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\n' +
        'void main(void){\n' +
        '    vec3  destColor = vec3(0.0);\n' +
        '    if(b_laplacian){\n' +
        '        vec2 offset[9];\n' +
        '        offset[0] = vec2(-1.0, -1.0);\n' +
        '        offset[1] = vec2( 0.0, -1.0);\n' +
        '        offset[2] = vec2( 1.0, -1.0);\n' +
        '        offset[3] = vec2(-1.0,  0.0);\n' +
        '        offset[4] = vec2( 0.0,  0.0);\n' +
        '        offset[5] = vec2( 1.0,  0.0);\n' +
        '        offset[6] = vec2(-1.0,  1.0);\n' +
        '        offset[7] = vec2( 0.0,  1.0);\n' +
        '        offset[8] = vec2( 1.0,  1.0);\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n\n' +
        '        destColor  += texture2D(texture, (fc + offset[0]) * Frag).rgb * coef[0];\n' +
        '        destColor  += texture2D(texture, (fc + offset[1]) * Frag).rgb * coef[1];\n' +
        '        destColor  += texture2D(texture, (fc + offset[2]) * Frag).rgb * coef[2];\n' +
        '        destColor  += texture2D(texture, (fc + offset[3]) * Frag).rgb * coef[3];\n' +
        '        destColor  += texture2D(texture, (fc + offset[4]) * Frag).rgb * coef[4];\n' +
        '        destColor  += texture2D(texture, (fc + offset[5]) * Frag).rgb * coef[5];\n' +
        '        destColor  += texture2D(texture, (fc + offset[6]) * Frag).rgb * coef[6];\n' +
        '        destColor  += texture2D(texture, (fc + offset[7]) * Frag).rgb * coef[7];\n' +
        '        destColor  += texture2D(texture, (fc + offset[8]) * Frag).rgb * coef[8];\n\n' +
        '        destColor =max(destColor, 0.0);\n' +
        '    }else{\n' +
        '        destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'laplacianFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'LIC-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n\n' +
        'uniform bool b_lic;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'uniform float sigma;\n\n' +
        'struct lic_t { \n' +
        '    vec2 p; \n' +
        '    vec2 t;\n' +
        '    float w;\n' +
        '    float dw;\n' +
        '};\n\n' +
        'void step(inout lic_t s) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 t = texture2D(tfm, s.p).xy;\n' +
        '    if (dot(t, s.t) < 0.0) t = -t;\n' +
        '    s.t = t;\n\n' +
        '    s.dw = (abs(t.x) > abs(t.y))? \n' +
        '        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n' +
        '        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n' +
        '    s.p += t * s.dw / src_size;\n' +
        '    s.w += s.dw;\n' +
        '}\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    float twoSigma2 = 2.0 * sigma * sigma;\n' +
        '    float halfWidth = 2.0 * sigma;\n' +
        '    vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / ' +
        'src_size.y);\n\n' +
        '    if(b_lic){\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        vec3 c = texture2D( src, uv ).xyz;\n' +
        '        float w = 1.0;\n\n' +
        '        lic_t a, b;\n' +
        '        a.p = b.p = uv;\n' +
        '        a.t = texture2D( tfm, uv ).xy / src_size;\n' +
        '        b.t = -a.t;\n' +
        '        a.w = b.w = 0.0;\n\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (a.w < halfWidth) {\n' +
        '                step(a);\n' +
        '                float k = a.dw * exp(-a.w * a.w / twoSigma2);\n' +
        '                c += k * texture2D(src, a.p).xyz;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n' +
        '            if (b.w < halfWidth) {\n' +
        '                step(b);\n' +
        '                float k = b.dw * exp(-b.w * b.w / twoSigma2);\n' +
        '                c += k * texture2D(src, b.p).xyz;\n' +
        '                w += k;\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '        }\n\n' +
        '        gl_FragColor = vec4(c / w, 1.0);\n' +
        '    }else{\n' +
        '        gl_FragColor = texture2D(src, uv);\n' +
        '    }\n\n' +
        '}\n',
    'LIC-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'luminance-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'uniform float threshold;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'const float redScale   = 0.298912;\n' +
        'const float greenScale = 0.586611;\n' +
        'const float blueScale  = 0.114478;\n' +
        'const vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\n' +
        'void main(void){\n\n' +
        '	vec4 smpColor = texture2D(texture, vec2(vTexCoord.s, 1.0 - vTexCoord.t));\n' +
        '	float luminance = dot(smpColor.rgb, monochromeScale);\n' +
        '	if(luminance<threshold){luminance = 0.0;}\n\n' +
        '	smpColor = vec4(vec3(luminance), 1.0);\n' +
        '	gl_FragColor =smpColor;\n' +
        '}\n',
    'luminance-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'phong-frag': 'precision mediump float;\n\n' +
        'uniform mat4 invMatrix;\n' +
        'uniform vec3 lightDirection;\n' +
        'uniform vec3 eyeDirection;\n' +
        'uniform vec4 ambientColor;\n' +
        'varying vec4 vColor;\n' +
        'varying vec3 vNormal;\n\n' +
        'void main(void){\n' +
        '	vec3 invLight = normalize(invMatrix*vec4(lightDirection,0.0)).xyz;\n' +
        '	vec3 invEye = normalize(invMatrix*vec4(eyeDirection,0.0)).xyz;\n' +
        '	vec3 halfLE = normalize(invLight+invEye);\n' +
        '	float diffuse = clamp(dot(vNormal,invLight),0.0,1.0);\n' +
        '	float specular = pow(clamp(dot(vNormal,halfLE),0.0,1.0),50.0);\n' +
        '	vec4 destColor = vColor * vec4(vec3(diffuse),1.0) + vec4(vec3(specular),1.0) + ' +
        'ambientColor;\n' +
        '	gl_FragColor = destColor;\n' +
        '}\n',
    'phong-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec3 normal;\n\n' +
        'uniform mat4 mvpMatrix;\n\n' +
        'varying vec4 vColor;\n' +
        'varying vec3 vNormal;\n\n' +
        'void main(void){\n' +
        '    vNormal = normal;\n' +
        '    vColor = color;\n' +
        '    gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'point-frag': 'precision mediump float;\n' +
        'varying vec4      vColor;\n\n' +
        'void main(void){\n' +
        '    gl_FragColor = vColor;\n' +
        '}\n',
    'point-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   float pointSize;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '    vColor        = color;\n' +
        '    gl_Position   = mvpMatrix * vec4(position, 1.0);\n' +
        '    gl_PointSize  = pointSize;\n' +
        '}\n',
    'pointLighting-frag': 'precision mediump float;\n\n' +
        'uniform mat4 invMatrix;\n' +
        'uniform vec3 lightPosition;\n' +
        'uniform vec3 eyeDirection;\n' +
        'uniform vec4 ambientColor;\n\n' +
        'varying vec4 vColor;\n' +
        'varying vec3 vNormal;\n' +
        'varying vec3 vPosition;\n\n' +
        'void main(void){\n' +
        '	vec3 lightVec = lightPosition -vPosition;\n' +
        '	vec3 invLight = normalize(invMatrix*vec4(lightVec,0.0)).xyz;\n' +
        '	vec3 invEye = normalize(invMatrix*vec4(eyeDirection,0.0)).xyz;\n' +
        '	vec3 halfLE = normalize(invLight+invEye);\n' +
        '	float diffuse = clamp(dot(vNormal,invLight),0.0,1.0);\n' +
        '	float specular = pow(clamp(dot(vNormal,halfLE),0.0,1.0),50.0);\n' +
        '	vec4 destColor = vColor * vec4(vec3(diffuse),1.0) + vec4(vec3(specular),1.0) + ' +
        'ambientColor;\n' +
        '	gl_FragColor = destColor;\n' +
        '}\n',
    'pointLighting-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec3 normal;\n\n' +
        'uniform mat4 mvpMatrix;\n' +
        'uniform mat4 mMatrix;\n\n' +
        'varying vec3 vPosition;\n' +
        'varying vec4 vColor;\n' +
        'varying vec3 vNormal;\n\n' +
        'void main(void){\n' +
        '    vPosition = (mMatrix*vec4(position,1.0)).xyz;\n' +
        '    vNormal = normal;\n' +
        '    vColor = color;\n' +
        '    gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'pointSprite-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'varying vec4      vColor;\n\n' +
        'void main(void){\n' +
        '    vec4 smpColor = vec4(1.0);\n' +
        '    smpColor = texture2D(texture,gl_PointCoord);\n' +
        '    if(smpColor.a == 0.0){\n' +
        '        discard;\n' +
        '    }else{\n' +
        '        gl_FragColor = vColor * smpColor;\n' +
        '    }\n' +
        '}\n',
    'pointSprite-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   float pointSize;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '    vColor        = color;\n' +
        '    gl_Position   = mvpMatrix * vec4(position, 1.0);\n' +
        '    gl_PointSize  = pointSize;\n' +
        '}\n',
    'projTexture-frag': 'precision mediump float;\n\n' +
        'uniform mat4      invMatrix;\n' +
        'uniform vec3      lightPosition;\n' +
        'uniform sampler2D texture;\n' +
        'varying vec3      vPosition;\n' +
        'varying vec3      vNormal;\n' +
        'varying vec4      vColor;\n' +
        'varying vec4      vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vec3  light    = lightPosition - vPosition;\n' +
        '	vec3  invLight = normalize(invMatrix * vec4(light, 0.0)).xyz;\n' +
        '	float diffuse  = clamp(dot(vNormal, invLight), 0.1, 1.0);\n' +
        '	vec4  smpColor = texture2DProj(texture, vTexCoord);\n' +
        '	gl_FragColor   = vColor * (0.5 + diffuse) * smpColor;\n' +
        '}\n',
    'projTexture-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 tMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec3 vPosition;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec4 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n' +
        '	vNormal     = normal;\n' +
        '	vColor      = color;\n' +
        '	vTexCoord   = tMatrix * vec4(vPosition, 1.0);\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'P_FDoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma_e;\n' +
        'uniform float sigma_r;\n' +
        'uniform float tau;\n\n' +
        'uniform bool b_FDoG;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 uv = gl_FragCoord.xy /src_size;\n' +
        '    if(b_FDoG){\n' +
        '        float twoSigmaESquared = 2.0 * sigma_e * sigma_e;\n' +
        '        float twoSigmaRSquared = 2.0 * sigma_r * sigma_r;\n\n' +
        '        vec2 t = texture2D(tfm, uv).xy;\n' +
        '        vec2 n = vec2(t.y, -t.x);\n' +
        '        vec2 nabs = abs(n);\n' +
        '        float ds = 1.0 / ((nabs.x > nabs.y)? nabs.x : nabs.y);\n' +
        '        n /= src_size;\n\n' +
        '        vec2 sum = texture2D( src, uv ).xx;\n' +
        '        vec2 norm = vec2(1.0, 1.0);\n\n' +
        '        float halfWidth = 2.0 * sigma_r;\n' +
        '        float d = ds;\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n\n' +
        '            if( d <= halfWidth) {\n' +
        '                vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n' +
        '                                    exp( -d * d / twoSigmaRSquared ));\n' +
        '                norm += 2.0 * kernel;\n\n' +
        '                vec2 L0 = texture2D( src, uv - d*n ).xx;\n' +
        '                vec2 L1 = texture2D( src, uv + d*n ).xx;\n' +
        '                sum += kernel * ( L0 + L1 );\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '            d+=ds;\n' +
        '        }\n\n' +
        '        sum /= norm;\n\n' +
        '        float diff = 100.0 * (sum.x - tau * sum.y);\n' +
        '        destColor= vec3(diff);\n' +
        '    }else{\n' +
        '        destColor = texture2D(src, uv).rgb;\n' +
        '    }\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'P_FDoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'P_FXDoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform sampler2D tfm;\n\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma;\n' +
        'uniform float k;\n' +
        'uniform float p;\n\n' +
        'uniform bool b_FXDoG;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 uv = gl_FragCoord.xy /src_size;\n' +
        '    if(b_FXDoG){\n' +
        '        float twoSigmaESquared = 2.0 * sigma * sigma;\n' +
        '        float twoSigmaRSquared = twoSigmaESquared * k * k;\n\n' +
        '        vec2 t = texture2D(tfm, uv).xy;\n' +
        '        vec2 n = vec2(t.y, -t.x);\n' +
        '        vec2 nabs = abs(n);\n' +
        '        float ds = 1.0 / ((nabs.x > nabs.y)? nabs.x : nabs.y);\n' +
        '        n /= src_size;\n\n' +
        '        vec2 sum = texture2D( src, uv ).xx;\n' +
        '        vec2 norm = vec2(1.0, 1.0);\n\n' +
        '        float halfWidth = 2.0 * sigma;\n' +
        '        float d = ds;\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n\n' +
        '            if( d <= halfWidth) {\n' +
        '                vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n' +
        '                                    exp( -d * d / twoSigmaRSquared ));\n' +
        '                norm += 2.0 * kernel;\n\n' +
        '                vec2 L0 = texture2D( src, uv - d*n ).xx;\n' +
        '                vec2 L1 = texture2D( src, uv + d*n ).xx;\n' +
        '                sum += kernel * ( L0 + L1 );\n' +
        '            }else{\n' +
        '                break;\n' +
        '            }\n' +
        '            d+=ds;\n' +
        '        }\n\n' +
        '        sum /= norm;\n\n' +
        '        float diff = 100.0 * ((1.0 + p) * sum.x - p * sum.y);\n' +
        '        destColor= vec3(diff);\n' +
        '    }else{\n' +
        '        destColor = texture2D(src, uv).rgb;\n' +
        '    }\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'P_FXDoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'refractionMapping-frag': 'precision mediump float;\n\n' +
        'uniform vec3        eyePosition;\n' +
        'uniform samplerCube cubeTexture;\n' +
        'uniform bool        refraction;\n' +
        'varying vec3        vPosition;\n' +
        'varying vec3        vNormal;\n' +
        'varying vec4        vColor;\n\n' +
        '//reflact calculation TODO\n' +
        '//vec3 egt_refract(vec3 p, vec3 n,float eta){\n' +
        '//}\n\n' +
        'void main(void){\n' +
        '	vec3 ref;\n' +
        '	if(refraction){\n' +
        '		ref = refract(normalize(vPosition - eyePosition), vNormal,0.6);\n' +
        '	}else{\n' +
        '		ref = vNormal;\n' +
        '	}\n' +
        '	vec4 envColor  = textureCube(cubeTexture, ref);\n' +
        '	vec4 destColor = vColor * envColor;\n' +
        '	gl_FragColor   = destColor;\n' +
        '}\n',
    'refractionMapping-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec3 vPosition;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n' +
        '	vNormal     = normalize((mMatrix * vec4(normal, 0.0)).xyz);\n' +
        '	vColor      = color;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'sepiaFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'uniform bool      sepia;\n' +
        'varying vec2      vTexCoord;\n\n' +
        'const float redScale   = 0.298912;\n' +
        'const float greenScale = 0.586611;\n' +
        'const float blueScale  = 0.114478;\n' +
        'const vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\n' +
        'const float sRedScale   = 1.07;\n' +
        'const float sGreenScale = 0.74;\n' +
        'const float sBlueScale  = 0.43;\n' +
        'const vec3  sepiaScale = vec3(sRedScale, sGreenScale, sBlueScale);\n\n' +
        'void main(void){\n' +
        '    vec4  smpColor  = texture2D(texture, vTexCoord);\n' +
        '    float grayColor = dot(smpColor.rgb, monochromeScale);\n\n' +
        '    vec3 monoColor = vec3(grayColor) * sepiaScale; \n' +
        '    smpColor = vec4(monoColor, 1.0);\n\n' +
        '    gl_FragColor = smpColor;\n' +
        '}\n',
    'sepiaFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'shadowDepthBuffer-frag': 'precision mediump float;\n\n' +
        'uniform bool depthBuffer;\n\n' +
        'varying vec4 vPosition;\n\n' +
        'vec4 convRGBA(float depth){\n' +
        '    float r = depth;\n' +
        '    float g = fract(r*255.0);\n' +
        '    float b = fract(g*255.0); \n' +
        '    float a = fract(b*255.0);\n' +
        '    float coef = 1.0/255.0;\n' +
        '    r-= g* coef; \n' +
        '    g-= b* coef; \n' +
        '    b-= a* coef; \n' +
        '    return vec4(r,g,b,a);\n' +
        '}\n\n' +
        'void main(void){\n' +
        '    vec4 convColor;\n' +
        '    if(depthBuffer){\n' +
        '        convColor = convRGBA(gl_FragCoord.z);\n' +
        '    }else{\n' +
        '        float near = 0.1;\n' +
        '        float far  = 150.0;\n' +
        '        float linerDepth = 1.0 / (far - near);\n' +
        '        linerDepth *= length(vPosition);\n' +
        '        convColor = convRGBA(linerDepth);\n' +
        '    }\n' +
        '    gl_FragColor = convColor;\n' +
        '}\n',
    'shadowDepthBuffer-vert': 'attribute vec3 position;\n' +
        'uniform mat4 mvpMatrix;\n\n' +
        'varying vec4 vPosition;\n\n' +
        'void main(void){\n' +
        '    vPosition = mvpMatrix * vec4(position, 1.0);\n' +
        '    gl_Position = vPosition;\n' +
        '}\n',
    'shadowScreen-frag': 'precision mediump float;\n\n' +
        'uniform mat4      invMatrix;\n' +
        'uniform vec3      lightPosition;\n' +
        'uniform sampler2D texture;\n' +
        'uniform bool      depthBuffer;\n' +
        'varying vec3      vPosition;\n' +
        'varying vec3      vNormal;\n' +
        'varying vec4      vColor;\n' +
        'varying vec4      vTexCoord;\n' +
        'varying vec4      vDepth;\n\n' +
        'float restDepth(vec4 RGBA){\n' +
        '    const float rMask = 1.0;\n' +
        '    const float gMask = 1.0 / 255.0;\n' +
        '    const float bMask = 1.0 / (255.0 * 255.0);\n' +
        '    const float aMask = 1.0 / (255.0 * 255.0 * 255.0);\n' +
        '    float depth = dot(RGBA, vec4(rMask, gMask, bMask, aMask));\n' +
        '    return depth;\n' +
        '}\n\n' +
        'void main(void){\n' +
        '    vec3  light     = lightPosition - vPosition;\n' +
        '    vec3  invLight  = normalize(invMatrix * vec4(light, 0.0)).xyz;\n' +
        '    float diffuse   = clamp(dot(vNormal, invLight), 0.1, 1.0);\n' +
        '    float shadow    = restDepth(texture2DProj(texture, vTexCoord));\n' +
        '    vec4 depthColor = vec4(1.0);\n' +
        '    if(vDepth.w > 0.0){\n' +
        '        if(depthBuffer){\n' +
        '            vec4 lightCoord = vDepth / vDepth.w;\n' +
        '            if(lightCoord.z - 0.0001 > shadow){\n' +
        '                depthColor  = vec4(0.5, 0.5, 0.5, 1.0);\n' +
        '            }\n' +
        '        }else{\n' +
        '            float near = 0.1;\n' +
        '            float far  = 150.0;\n' +
        '            float linerDepth = 1.0 / (far - near);\n' +
        '            linerDepth *= length(vPosition.xyz - lightPosition);\n' +
        '            if(linerDepth - 0.0001 > shadow){\n' +
        '                depthColor  = vec4(0.5, 0.5, 0.5, 1.0);\n' +
        '            }\n' +
        '        }\n' +
        '    }\n' +
        '    gl_FragColor = vColor * (vec3(diffuse),1.0) * depthColor;\n' +
        '}\n',
    'shadowScreen-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mMatrix;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 tMatrix;\n' +
        'uniform   mat4 lgtMatrix;\n' +
        'varying   vec3 vPosition;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec4 vTexCoord;\n' +
        'varying   vec4 vDepth;\n\n' +
        'void main(void){\n' +
        '    vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n' +
        '    vNormal     = normal;\n' +
        '    vColor      = color;\n' +
        '    vTexCoord   = tMatrix * vec4(vPosition, 1.0);\n' +
        '    vDepth      = lgtMatrix * vec4(position, 1.0);\n' +
        '    gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'sobelFilter-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n\n' +
        'uniform bool b_sobel;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'uniform float hCoef[9];\n' +
        'uniform float vCoef[9];\n' +
        'varying vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    if(b_sobel){\n' +
        '        vec2 offset[9];\n' +
        '        offset[0] = vec2(-1.0, -1.0);\n' +
        '        offset[1] = vec2( 0.0, -1.0);\n' +
        '        offset[2] = vec2( 1.0, -1.0);\n' +
        '        offset[3] = vec2(-1.0,  0.0);\n' +
        '        offset[4] = vec2( 0.0,  0.0);\n' +
        '        offset[5] = vec2( 1.0,  0.0);\n' +
        '        offset[6] = vec2(-1.0,  1.0);\n' +
        '        offset[7] = vec2( 0.0,  1.0);\n' +
        '        offset[8] = vec2( 1.0,  1.0);\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n' +
        '        vec3  horizonColor = vec3(0.0);\n' +
        '        vec3  verticalColor = vec3(0.0);\n\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[0]) * Frag).rgb * hCoef' +
        '[0];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[1]) * Frag).rgb * hCoef' +
        '[1];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[2]) * Frag).rgb * hCoef' +
        '[2];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[3]) * Frag).rgb * hCoef' +
        '[3];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[4]) * Frag).rgb * hCoef' +
        '[4];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[5]) * Frag).rgb * hCoef' +
        '[5];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[6]) * Frag).rgb * hCoef' +
        '[6];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[7]) * Frag).rgb * hCoef' +
        '[7];\n' +
        '        horizonColor  += texture2D(texture, (fc + offset[8]) * Frag).rgb * hCoef' +
        '[8];\n\n' +
        '        verticalColor += texture2D(texture, (fc + offset[0]) * Frag).rgb * vCoef' +
        '[0];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[1]) * Frag).rgb * vCoef' +
        '[1];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[2]) * Frag).rgb * vCoef' +
        '[2];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[3]) * Frag).rgb * vCoef' +
        '[3];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[4]) * Frag).rgb * vCoef' +
        '[4];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[5]) * Frag).rgb * vCoef' +
        '[5];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[6]) * Frag).rgb * vCoef' +
        '[6];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[7]) * Frag).rgb * vCoef' +
        '[7];\n' +
        '        verticalColor += texture2D(texture, (fc + offset[8]) * Frag).rgb * vCoef' +
        '[8];\n' +
        '        destColor = vec3(sqrt(horizonColor * horizonColor + verticalColor * vert' +
        'icalColor));\n' +
        '    }else{\n' +
        '        destColor = texture2D(texture, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'sobelFilter-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'specCpt-frag': 'precision mediump float;\n\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'specCpt-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 invMatrix;\n' +
        'uniform   vec3 lightDirection;\n' +
        'uniform   vec3 eyeDirection;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n' +
        '	vec3  invEye   = normalize(invMatrix * vec4(eyeDirection, 0.0)).xyz;\n' +
        '	vec3  halfLE   = normalize(invLight + invEye);\n' +
        '	float specular = pow(clamp(dot(normal, halfLE), 0.0, 1.0), 50.0);\n' +
        '	vColor         = color * vec4(vec3(specular), 1.0);\n' +
        '	gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'specular-frag': 'precision mediump float;\n\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	gl_FragColor = vColor;\n' +
        '}\n',
    'specular-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec3 normal;\n\n' +
        'uniform mat4 mvpMatrix;\n' +
        'uniform mat4 invMatrix;\n\n' +
        'uniform vec3 lightDirection;\n' +
        'uniform vec3 eyeDirection;\n' +
        'uniform vec4 ambientColor;\n' +
        'varying vec4 vColor;\n\n' +
        'void main(void){\n' +
        '    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0.0)).xyz;\n' +
        '    vec3 invEye = normalize(invMatrix* vec4(eyeDirection,0.0)).xyz;\n' +
        '    vec3 halfLE = normalize(invLight+invEye);\n\n' +
        '    float diffuse = clamp(dot(invLight,normal),0.0,1.0);\n' +
        '    float specular = pow(clamp(dot(normal,halfLE),0.0,1.0),50.0);\n' +
        '    vec4 light = color*vec4(vec3(diffuse),1.0)+vec4(vec3(specular),1.0);\n' +
        '    vColor = light + ambientColor;\n' +
        '    gl_Position    = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'SST-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'void main (void) {\n' +
        '    vec2 src_size = vec2(cvsWidth, cvsHeight);\n' +
        '    vec2 uv = gl_FragCoord.xy / src_size;\n' +
        '    vec2 d = 1.0 / src_size;\n' +
        '    vec3 u = (\n' +
        '        -1.0 * texture2D(src, uv + vec2(-d.x, -d.y)).xyz +\n' +
        '        -2.0 * texture2D(src, uv + vec2(-d.x,  0.0)).xyz + \n' +
        '        -1.0 * texture2D(src, uv + vec2(-d.x,  d.y)).xyz +\n' +
        '        +1.0 * texture2D(src, uv + vec2( d.x, -d.y)).xyz +\n' +
        '        +2.0 * texture2D(src, uv + vec2( d.x,  0.0)).xyz + \n' +
        '        +1.0 * texture2D(src, uv + vec2( d.x,  d.y)).xyz\n' +
        '        ) / 4.0;\n\n' +
        '    vec3 v = (\n' +
        '           -1.0 * texture2D(src, uv + vec2(-d.x, -d.y)).xyz + \n' +
        '           -2.0 * texture2D(src, uv + vec2( 0.0, -d.y)).xyz + \n' +
        '           -1.0 * texture2D(src, uv + vec2( d.x, -d.y)).xyz +\n' +
        '           +1.0 * texture2D(src, uv + vec2(-d.x,  d.y)).xyz +\n' +
        '           +2.0 * texture2D(src, uv + vec2( 0.0,  d.y)).xyz + \n' +
        '           +1.0 * texture2D(src, uv + vec2( d.x,  d.y)).xyz\n' +
        '           ) / 4.0;\n\n' +
        '    gl_FragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);\n' +
        '}\n',
    'SST-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'stencilBufferOutline-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'uniform bool      useTexture;\n' +
        'varying vec4      vColor;\n' +
        'varying vec2      vTextureCoord;\n\n' +
        'void main(void){\n' +
        '	vec4 smpColor = vec4(1.0);\n' +
        '	if(useTexture){\n' +
        '		smpColor = texture2D(texture, vTextureCoord);\n' +
        '	}\n' +
        '	gl_FragColor = vColor * smpColor;\n' +
        '}\n',
    'stencilBufferOutline-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'attribute vec2 textureCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   mat4 invMatrix;\n' +
        'uniform   vec3 lightDirection;\n' +
        'uniform   bool useLight;\n' +
        'uniform   bool outline;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec2 vTextureCoord;\n\n' +
        'void main(void){\n' +
        '	if(useLight){\n' +
        '		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n' +
        '		float diffuse  = clamp(dot(normal, invLight), 0.1, 1.0);\n' +
        '		vColor         = color * vec4(vec3(diffuse), 1.0);\n' +
        '	}else{\n' +
        '		vColor         = color;\n' +
        '	}\n' +
        '	vTextureCoord      = textureCoord;\n' +
        '	vec3 oPosition     = position;\n' +
        '	if(outline){\n' +
        '		oPosition     += normal * 0.1;\n' +
        '	}\n' +
        '	gl_Position = mvpMatrix * vec4(oPosition, 1.0);\n' +
        '}\n',
    'synth-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture1;\n' +
        'uniform sampler2D texture2;\n' +
        'uniform bool      glare;\n' +
        'varying vec2      vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vec4  destColor = texture2D(texture1, vTexCoord);\n' +
        '	vec4  smpColor  = texture2D(texture2, vec2(vTexCoord.s, 1.0 - vTexCoord.t));\n' +
        '	if(glare){\n' +
        '		destColor += smpColor * 0.4;\n' +
        '	}\n' +
        '	gl_FragColor = destColor;\n' +
        '}\n',
    'synth-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'texture-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D texture;\n' +
        'varying vec4      vColor;\n' +
        'varying vec2      vTextureCoord;\n\n' +
        'void main(void){\n' +
        '    vec4 smpColor = texture2D(texture, vTextureCoord);\n' +
        '    gl_FragColor  = vColor * smpColor;\n' +
        '}\n',
    'texture-vert': 'attribute vec3 position;\n' +
        'attribute vec4 color;\n' +
        'attribute vec2 textureCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec4 vColor;\n' +
        'varying   vec2 vTextureCoord;\n\n' +
        'void main(void){\n' +
        '    vColor        = color;\n' +
        '    vTextureCoord = textureCoord;\n' +
        '    gl_Position   = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'TF-frag': '// Tangent Field\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n' +
        'uniform float hCoef[9];\n' +
        'uniform float vCoef[9];\n\n' +
        'const float redScale   = 0.298912;\n' +
        'const float greenScale = 0.586611;\n' +
        'const float blueScale  = 0.114478;\n' +
        'const vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\n' +
        'void main (void) {\n' +
        '    vec2 offset[9];\n' +
        '    offset[0] = vec2(-1.0, -1.0);\n' +
        '    offset[1] = vec2( 0.0, -1.0);\n' +
        '    offset[2] = vec2( 1.0, -1.0);\n' +
        '    offset[3] = vec2(-1.0,  0.0);\n' +
        '    offset[4] = vec2( 0.0,  0.0);\n' +
        '    offset[5] = vec2( 1.0,  0.0);\n' +
        '    offset[6] = vec2(-1.0,  1.0);\n' +
        '    offset[7] = vec2( 0.0,  1.0);\n' +
        '    offset[8] = vec2( 1.0,  1.0);\n' +
        '    float tFrag = 1.0 / cvsHeight;\n' +
        '    float sFrag = 1.0 / cvsWidth;\n' +
        '    vec2  Frag = vec2(sFrag,tFrag);\n' +
        '    vec2  uv = vec2(gl_FragCoord.s, gl_FragCoord.t);\n' +
        '    float  horizonColor = 0.0;\n' +
        '    float  verticalColor = 0.0;\n\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[0]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[0];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[1]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[1];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[2]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[2];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[3]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[3];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[4]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[4];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[5]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[5];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[6]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[6];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[7]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[7];\n' +
        '    horizonColor  += dot(texture2D(src, (uv + offset[8]) * Frag).rgb, monochrome' +
        'Scale) * hCoef[8];\n\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[0]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[0];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[1]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[1];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[2]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[2];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[3]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[3];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[4]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[4];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[5]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[5];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[6]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[6];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[7]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[7];\n' +
        '    verticalColor += dot(texture2D(src, (uv + offset[8]) * Frag).rgb, monochrome' +
        'Scale) * vCoef[8];\n\n' +
        '    float mag = sqrt(horizonColor * horizonColor + verticalColor * verticalColor' +
        ');\n' +
        '    float vx = verticalColor/mag;\n' +
        '    float vy = horizonColor/mag;\n\n' +
        '    gl_FragColor = vec4(vx, vy, mag, 1.0);\n' +
        '}\n',
    'TF-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'TFM-frag': '// by Jan Eric Kyprianidis <www.kyprianidis.com>\n' +
        'precision mediump float;\n\n' +
        'uniform sampler2D src;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'void main (void) {\n' +
        '    vec2 uv = gl_FragCoord.xy / vec2(cvsWidth, cvsHeight);\n' +
        '    vec3 g = texture2D(src, uv).xyz;\n\n' +
        '    float lambda1 = 0.5 * (g.y + g.x + sqrt(g.y*g.y - 2.0*g.x*g.y + g.x*g.x + 4.' +
        '0*g.z*g.z));\n' +
        '    float lambda2 = 0.5 * (g.y + g.x - sqrt(g.y*g.y - 2.0*g.x*g.y + g.x*g.x + 4.' +
        '0*g.z*g.z));\n\n' +
        '    vec2 v = vec2(lambda1 - g.x, -g.z);\n' +
        '    vec2 t;\n' +
        '    if (length(v) > 0.0) { \n' +
        '        t = normalize(v);\n' +
        '    } else {\n' +
        '        t = vec2(0.0, 1.0);\n' +
        '    }\n\n' +
        '    float phi = atan(t.y, t.x);\n\n' +
        '    float A = (lambda1 + lambda2 > 0.0)?(lambda1 - lambda2) / (lambda1 + lambda2' +
        ') : 0.0;\n' +
        '    gl_FragColor = vec4(t, phi, A);\n' +
        '}\n',
    'TFM-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n',
    'toonShading-frag': 'precision mediump float;\n\n' +
        'uniform mat4      invMatrix;\n' +
        'uniform vec3      lightDirection;\n' +
        'uniform sampler2D texture;\n' +
        'uniform vec4      edgeColor;\n' +
        'varying vec3      vNormal;\n' +
        'varying vec4      vColor;\n\n' +
        'void main(void){\n' +
        '	if(edgeColor.a > 0.0){\n' +
        '		gl_FragColor   = edgeColor;\n' +
        '	}else{\n' +
        '		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n' +
        '		float diffuse  = clamp(dot(vNormal, invLight), 0.1, 1.0);\n' +
        '		vec4  smpColor = texture2D(texture, vec2(diffuse, 0.0));\n' +
        '		gl_FragColor   = vColor * smpColor;\n' +
        '	}\n' +
        '}\n',
    'toonShading-vert': 'attribute vec3 position;\n' +
        'attribute vec3 normal;\n' +
        'attribute vec4 color;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'uniform   bool edge;\n' +
        'varying   vec3 vNormal;\n' +
        'varying   vec4 vColor;\n\n' +
        'void main(void){\n' +
        '	vec3 pos    = position;\n' +
        '	if(edge){\n' +
        '		pos    += normal * 0.05;\n' +
        '	}\n' +
        '	vNormal     = normal;\n' +
        '	vColor      = color;\n' +
        '	gl_Position = mvpMatrix * vec4(pos, 1.0);\n' +
        '}\n',
    'XDoG-frag': 'precision mediump float;\n\n' +
        'uniform sampler2D src;\n\n' +
        'uniform bool b_XDoG;\n' +
        'uniform float cvsHeight;\n' +
        'uniform float cvsWidth;\n\n' +
        'uniform float sigma;\n' +
        'uniform float k;\n' +
        'uniform float p;\n' +
        'uniform float epsilon;\n' +
        'uniform float phi;\n' +
        'varying vec2 vTexCoord;\n\n' +
        'float cosh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float cosH = (tmp + 1.0 / tmp) / 2.0;\n' +
        '    return cosH;\n' +
        '}\n\n' +
        'float tanh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n' +
        '    return tanH;\n' +
        '}\n\n' +
        'float sinh(float val)\n' +
        '{\n' +
        '    float tmp = exp(val);\n' +
        '    float sinH = (tmp - 1.0 / tmp) / 2.0;\n' +
        '    return sinH;\n' +
        '}\n\n' +
        'void main(void){\n' +
        '    vec3 destColor = vec3(0.0);\n' +
        '    if(b_XDoG){\n' +
        '        float tFrag = 1.0 / cvsHeight;\n' +
        '        float sFrag = 1.0 / cvsWidth;\n' +
        '        vec2  Frag = vec2(sFrag,tFrag);\n' +
        '        vec2 uv = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n' +
        '        float twoSigmaESquared = 2.0 * sigma * sigma;\n' +
        '        float twoSigmaRSquared = twoSigmaESquared * k * k;\n' +
        '        int halfWidth = int(ceil( 1.0 * sigma * k ));\n\n' +
        '        const int MAX_NUM_ITERATION = 99999;\n' +
        '        vec2 sum = vec2(0.0);\n' +
        '        vec2 norm = vec2(0.0);\n\n' +
        '        for(int cnt=0;cnt<MAX_NUM_ITERATION;cnt++){\n' +
        '            if(cnt > (2*halfWidth+1)*(2*halfWidth+1)){break;}\n' +
        '            int i = int(cnt / (2*halfWidth+1)) - halfWidth;\n' +
        '            int j = cnt - halfWidth - int(cnt / (2*halfWidth+1)) * (2*halfWidth+' +
        '1);\n\n' +
        '            float d = length(vec2(i,j));\n' +
        '            vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n' +
        '                                exp( -d * d / twoSigmaRSquared ));\n\n' +
        '            vec2 L = texture2D(src, (uv + vec2(i,j)) * Frag).xx;\n\n' +
        '            norm += kernel;\n' +
        '            sum += kernel * L;\n' +
        '        }\n\n' +
        '        sum /= norm;\n\n' +
        '        float H = 100.0 * ((1.0 + p) * sum.x - p * sum.y);\n' +
        '        float edge = ( H > epsilon )? 1.0 : 1.0 + tanh( phi * (H - epsilon));\n' +
        '        destColor = vec3(edge);\n' +
        '    }else{\n' +
        '        destColor = texture2D(src, vTexCoord).rgb;\n' +
        '    }\n\n' +
        '    gl_FragColor = vec4(destColor, 1.0);\n' +
        '}\n',
    'XDoG-vert': 'attribute vec3 position;\n' +
        'attribute vec2 texCoord;\n' +
        'uniform   mat4 mvpMatrix;\n' +
        'varying   vec2 vTexCoord;\n\n' +
        'void main(void){\n' +
        '	vTexCoord   = texCoord;\n' +
        '	gl_Position = mvpMatrix * vec4(position, 1.0);\n' +
        '}\n'
};
/* =========================================================================
 *
 *  webgl_model.ts
 *  simple 3d model for webgl
 *
 * ========================================================================= */
/// <reference path="./cv_colorSpace.ts" />
var EcognitaMathLib;
(function (EcognitaMathLib) {
    var BoardModel = /** @class */ (function () {
        function BoardModel(u_position, u_color, need_normal, need_color, need_texCoord) {
            if (u_position === void 0) { u_position = undefined; }
            if (u_color === void 0) { u_color = undefined; }
            if (need_normal === void 0) { need_normal = true; }
            if (need_color === void 0) { need_color = true; }
            if (need_texCoord === void 0) { need_texCoord = false; }
            this.data = new Array();
            var position = [
                -1.0, 0.0, -1.0,
                1.0, 0.0, -1.0,
                -1.0, 0.0, 1.0,
                1.0, 0.0, 1.0
            ];
            var normal = [
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0
            ];
            this.index = [
                0, 1, 2,
                3, 2, 1
            ];
            var texCoord = [
                0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0,
                1.0, 1.0
            ];
            for (var i = 0; i < 4; i++) {
                if (u_position == undefined)
                    this.data.push(position[i * 3 + 0], position[i * 3 + 1], position[i * 3 + 2]);
                else
                    this.data.push(u_position[i * 3 + 0], u_position[i * 3 + 1], u_position[i * 3 + 2]);
                if (need_normal)
                    this.data.push(normal[i * 3 + 0], normal[i * 3 + 1], normal[i * 3 + 2]);
                if (u_color == undefined) {
                    var color = [
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0
                    ];
                    if (need_color)
                        this.data.push(color[i * 4 + 0], color[i * 4 + 1], color[i * 4 + 2], color[i * 4 + 3]);
                }
                else {
                    if (need_color)
                        this.data.push(u_color[i * 4 + 0], u_color[i * 4 + 1], u_color[i * 4 + 2], u_color[i * 4 + 3]);
                }
                if (need_texCoord)
                    this.data.push(texCoord[i * 2 + 0], texCoord[i * 2 + 1]);
            }
        }
        return BoardModel;
    }());
    EcognitaMathLib.BoardModel = BoardModel;
    var TorusModel = /** @class */ (function () {
        function TorusModel(vcrs, hcrs, vr, hr, color, need_normal, need_texture) {
            if (need_texture === void 0) { need_texture = false; }
            this.verCrossSectionSmooth = vcrs;
            this.horCrossSectionSmooth = hcrs;
            this.verRadius = vr;
            this.horRadius = hr;
            this.data = new Array();
            this.index = new Array();
            this.normal = new Array();
            this.preCalculate(color, need_normal, need_texture);
        }
        TorusModel.prototype.preCalculate = function (color, need_normal, need_texture) {
            if (need_texture === void 0) { need_texture = false; }
            //calculate pos and col
            for (var i = 0; i <= this.verCrossSectionSmooth; i++) {
                var verIncrement = Math.PI * 2 / this.verCrossSectionSmooth * i;
                var verX = Math.cos(verIncrement);
                var verY = Math.sin(verIncrement);
                for (var ii = 0; ii <= this.horCrossSectionSmooth; ii++) {
                    var horIncrement = Math.PI * 2 / this.horCrossSectionSmooth * ii;
                    var horX = (verX * this.verRadius + this.horRadius) * Math.cos(horIncrement);
                    var horY = verY * this.verRadius;
                    var horZ = (verX * this.verRadius + this.horRadius) * Math.sin(horIncrement);
                    this.data.push(horX, horY, horZ);
                    if (need_normal) {
                        var nx = verX * Math.cos(horIncrement);
                        var nz = verX * Math.sin(horIncrement);
                        this.normal.push(nx, verY, nz);
                        this.data.push(nx, verY, nz);
                    }
                    //hsv2rgb
                    if (color == undefined) {
                        var rgba = EcognitaMathLib.HSV2RGB(360 / this.horCrossSectionSmooth * ii, 1, 1, 1);
                        this.data.push(rgba[0], rgba[1], rgba[2], rgba[3]);
                    }
                    else {
                        this.data.push(color[0], color[1], color[2], color[3]);
                    }
                    if (need_texture) {
                        var rs = 1 / this.horCrossSectionSmooth * ii;
                        var rt = 1 / this.verCrossSectionSmooth * i + 0.5;
                        if (rt > 1.0) {
                            rt -= 1.0;
                        }
                        rt = 1.0 - rt;
                        this.data.push(rs, rt);
                    }
                }
            }
            //calculate index
            for (i = 0; i < this.verCrossSectionSmooth; i++) {
                for (ii = 0; ii < this.horCrossSectionSmooth; ii++) {
                    verIncrement = (this.horCrossSectionSmooth + 1) * i + ii;
                    this.index.push(verIncrement, verIncrement + this.horCrossSectionSmooth + 1, verIncrement + 1);
                    this.index.push(verIncrement + this.horCrossSectionSmooth + 1, verIncrement + this.horCrossSectionSmooth + 2, verIncrement + 1);
                }
            }
        };
        return TorusModel;
    }());
    EcognitaMathLib.TorusModel = TorusModel;
    var ShpereModel = /** @class */ (function () {
        function ShpereModel(vcrs, hcrs, rad, color, need_normal, need_texture) {
            if (need_texture === void 0) { need_texture = false; }
            this.verCrossSectionSmooth = vcrs;
            this.horCrossSectionSmooth = hcrs;
            this.Radius = rad;
            this.data = new Array();
            this.index = new Array();
            this.preCalculate(color, need_normal, need_texture);
        }
        ShpereModel.prototype.preCalculate = function (color, need_normal, need_texture) {
            if (need_texture === void 0) { need_texture = false; }
            //calculate pos and col
            for (var i = 0; i <= this.verCrossSectionSmooth; i++) {
                var verIncrement = Math.PI / this.verCrossSectionSmooth * i;
                var verX = Math.cos(verIncrement);
                var verY = Math.sin(verIncrement);
                for (var ii = 0; ii <= this.horCrossSectionSmooth; ii++) {
                    var horIncrement = Math.PI * 2 / this.horCrossSectionSmooth * ii;
                    var horX = verY * this.Radius * Math.cos(horIncrement);
                    var horY = verX * this.Radius;
                    var horZ = verY * this.Radius * Math.sin(horIncrement);
                    this.data.push(horX, horY, horZ);
                    if (need_normal) {
                        var nx = verY * Math.cos(horIncrement);
                        var nz = verY * Math.sin(horIncrement);
                        this.data.push(nx, verX, nz);
                    }
                    //hsv2rgb
                    if (color == undefined) {
                        var rgba = EcognitaMathLib.HSV2RGB(360 / this.horCrossSectionSmooth * i, 1, 1, 1);
                        this.data.push(rgba[0], rgba[1], rgba[2], rgba[3]);
                    }
                    else {
                        this.data.push(color[0], color[1], color[2], color[3]);
                    }
                    if (need_texture) {
                        this.data.push(1 - 1 / this.horCrossSectionSmooth * ii, 1 / this.verCrossSectionSmooth * i);
                    }
                }
            }
            //calculate index
            for (i = 0; i < this.verCrossSectionSmooth; i++) {
                for (ii = 0; ii < this.horCrossSectionSmooth; ii++) {
                    verIncrement = (this.horCrossSectionSmooth + 1) * i + ii;
                    this.index.push(verIncrement, verIncrement + 1, verIncrement + this.horCrossSectionSmooth + 2);
                    this.index.push(verIncrement, verIncrement + this.horCrossSectionSmooth + 2, verIncrement + this.horCrossSectionSmooth + 1);
                }
            }
        };
        return ShpereModel;
    }());
    EcognitaMathLib.ShpereModel = ShpereModel;
    var CubeModel = /** @class */ (function () {
        function CubeModel(side, color, need_normal, need_texture) {
            if (need_texture === void 0) { need_texture = false; }
            this.side = side;
            this.data = new Array();
            this.index = [
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
                12, 13, 14, 12, 14, 15,
                16, 17, 18, 16, 18, 19,
                20, 21, 22, 20, 22, 23
            ];
            var hs = side * 0.5;
            var pos = [
                -hs, -hs, hs, hs, -hs, hs, hs, hs, hs, -hs, hs, hs,
                -hs, -hs, -hs, -hs, hs, -hs, hs, hs, -hs, hs, -hs, -hs,
                -hs, hs, -hs, -hs, hs, hs, hs, hs, hs, hs, hs, -hs,
                -hs, -hs, -hs, hs, -hs, -hs, hs, -hs, hs, -hs, -hs, hs,
                hs, -hs, -hs, hs, hs, -hs, hs, hs, hs, hs, -hs, hs,
                -hs, -hs, -hs, -hs, -hs, hs, -hs, hs, hs, -hs, hs, -hs
            ];
            var normal = [
                -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
                -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
                -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
                -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
                1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0
            ];
            var col = new Array();
            for (var i = 0; i < pos.length / 3; i++) {
                if (color != undefined) {
                    var tc = color;
                }
                else {
                    tc = EcognitaMathLib.HSV2RGB(360 / pos.length / 3 * i, 1, 1, 1);
                }
                col.push(tc[0], tc[1], tc[2], tc[3]);
            }
            var st = [
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0
            ];
            var cubeVertexNum = 24;
            for (var i = 0; i < cubeVertexNum; i++) {
                //pos
                this.data.push(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]);
                //normal
                if (need_normal) {
                    this.data.push(normal[i * 3 + 0], normal[i * 3 + 1], normal[i * 3 + 2]);
                }
                //color
                this.data.push(col[i * 4 + 0], col[i * 4 + 1], col[i * 4 + 2], col[i * 4 + 3]);
                //texture
                if (need_texture) {
                    this.data.push(st[i * 2 + 0], st[i * 2 + 1]);
                }
            }
        }
        return CubeModel;
    }());
    EcognitaMathLib.CubeModel = CubeModel;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var Utils;
(function (Utils) {
    ;
    var HashSet = /** @class */ (function () {
        function HashSet() {
            this.items = {};
        }
        HashSet.prototype.set = function (key, value) {
            this.items[key] = value;
        };
        HashSet.prototype.delete = function (key) {
            return delete this.items[key];
        };
        HashSet.prototype.has = function (key) {
            return key in this.items;
        };
        HashSet.prototype.get = function (key) {
            return this.items[key];
        };
        HashSet.prototype.len = function () {
            return Object.keys(this.items).length;
        };
        HashSet.prototype.forEach = function (f) {
            for (var k in this.items) {
                f(k, this.items[k]);
            }
        };
        return HashSet;
    }());
    Utils.HashSet = HashSet;
})(Utils || (Utils = {}));
/// <reference path="../lib/HashSet.ts" />
var Utils;
(function (Utils) {
    var FilterViewerUI = /** @class */ (function () {
        function FilterViewerUI(data) {
            var _this = this;
            this.gui = new dat.gui.GUI();
            this.data = data;
            this.gui.remember(data);
            this.uiController = new Utils.HashSet();
            this.folderHashSet = new Utils.HashSet();
            this.folderHashSet.set("f", "Filter");
            //get all folder name
            this.folderName = [];
            this.folderHashSet.forEach(function (k, v) {
                _this.folderName.push(k);
            });
            this.initData();
            this.initFolder();
        }
        FilterViewerUI.prototype.initFolder = function () {
            var _this = this;
            this.folderName.forEach(function (fn) {
                var f = _this.gui.addFolder(_this.folderHashSet.get(fn));
                for (var key in _this.data) {
                    //judge this key is in folder or not
                    var f_name = key.split("_");
                    if (key.includes('_') && f_name[0] == fn) {
                        var c = f.add(_this.data, key).listen();
                        _this.uiController.set(key, c);
                    }
                }
            });
        };
        FilterViewerUI.prototype.initData = function () {
            for (var key in this.data) {
                if (!key.includes('_')) {
                    this.gui.add(this.data, key);
                }
            }
        };
        return FilterViewerUI;
    }());
    Utils.FilterViewerUI = FilterViewerUI;
})(Utils || (Utils = {}));
/* =========================================================================
 *
 *  EgnWebGL.ts
 *  construct a webgl environment
 *  v0.1
 *
 * ========================================================================= */
/// <reference path="../../../../lib_webgl/ts_scripts/lib/cv_imread.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/cv_colorSpace.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/extra_utils.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/webgl_matrix.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/webgl_quaternion.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/webgl_utils.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/webgl_shaders.ts" />
/// <reference path="../../../../lib_webgl/ts_scripts/lib/webgl_model.ts" />
/// <reference path="../lib/HashSet.ts" />
/// <reference path="../lib/EgnFilterViewerUI.ts" />
var EcognitaWeb3D;
(function (EcognitaWeb3D) {
    var WebGLEnv = /** @class */ (function () {
        function WebGLEnv(cvs) {
            this.chkWebGLEnv(cvs);
        }
        WebGLEnv.prototype.loadTexture = function (file_name, isFloat, glType, glInterType, useMipmap, channel) {
            var _this = this;
            if (isFloat === void 0) { isFloat = false; }
            if (glType === void 0) { glType = gl.CLAMP_TO_EDGE; }
            if (glInterType === void 0) { glInterType = gl.LINEAR; }
            if (useMipmap === void 0) { useMipmap = true; }
            if (channel === void 0) { channel = 4; }
            var tex = null;
            var image = EcognitaMathLib.imread(file_name);
            image.onload = (function () {
                tex = new EcognitaMathLib.WebGL_Texture(channel, isFloat, image, glType, glInterType, useMipmap);
                _this.Texture.set(file_name, tex);
            });
        };
        WebGLEnv.prototype.chkWebGLEnv = function (cvs) {
            this.canvas = cvs;
            try {
                gl = this.canvas.getContext("webgl") || this.canvas.getContext("experimental-webgl");
                this.stats = new Stats();
                document.body.appendChild(this.stats.dom);
            }
            catch (e) { }
            if (!gl)
                throw new Error("Could not initialise WebGL");
            //check extension
            var ext = gl.getExtension('OES_texture_float');
            if (ext == null) {
                throw new Error("float texture not supported");
            }
        };
        WebGLEnv.prototype.initGlobalVariables = function () {
            this.vbo = new Array();
            this.ibo = new Array();
            this.Texture = new Utils.HashSet();
            this.matUtil = new EcognitaMathLib.WebGLMatrix();
            this.quatUtil = new EcognitaMathLib.WebGLQuaternion();
        };
        WebGLEnv.prototype.initGlobalMatrix = function () {
            this.MATRIX = new Utils.HashSet();
            var m = this.matUtil;
            this.MATRIX.set("mMatrix", m.identity(m.create()));
            this.MATRIX.set("vMatrix", m.identity(m.create()));
            this.MATRIX.set("pMatrix", m.identity(m.create()));
            this.MATRIX.set("vpMatrix", m.identity(m.create()));
            this.MATRIX.set("mvpMatrix", m.identity(m.create()));
            this.MATRIX.set("invMatrix", m.identity(m.create()));
        };
        WebGLEnv.prototype.loadExtraLibrary = function (ui_data) {
            this.ui_data = ui_data;
            //load extral library
            this.uiUtil = new Utils.FilterViewerUI(this.ui_data);
            this.extHammer = new EcognitaMathLib.Hammer_Utils(this.canvas);
        };
        WebGLEnv.prototype.loadInternalLibrary = function (shaderlist) {
            var _this = this;
            //load internal library
            this.framebuffers = new Utils.HashSet();
            //init shaders and uniLocations
            this.shaders = new Utils.HashSet();
            this.uniLocations = new Utils.HashSet();
            shaderlist.forEach(function (s) {
                var shader = new EcognitaMathLib.WebGL_Shader(Shaders, s.name + "-vert", s.name + "-frag");
                _this.shaders.set(s.name, shader);
                _this.uniLocations.set(s.name, new Array());
            });
        };
        WebGLEnv.prototype.settingFrameBuffer = function (frameBufferName) {
            //frame buffer
            var fBufferWidth = this.canvas.width;
            var fBufferHeight = this.canvas.height;
            var frameBuffer = new EcognitaMathLib.WebGL_FrameBuffer(fBufferWidth, fBufferHeight);
            frameBuffer.bindFrameBuffer();
            frameBuffer.bindDepthBuffer();
            //frameBuffer.renderToShadowTexure();
            frameBuffer.renderToFloatTexure();
            frameBuffer.release();
            this.framebuffers.set(frameBufferName, frameBuffer);
        };
        WebGLEnv.prototype.renderSceneByFrameBuffer = function (framebuffer, func, texid) {
            if (texid === void 0) { texid = gl.TEXTURE0; }
            framebuffer.bindFrameBuffer();
            func();
            gl.activeTexture(texid);
            gl.bindTexture(gl.TEXTURE_2D, framebuffer.targetTexture);
        };
        WebGLEnv.prototype.renderBoardByFrameBuffer = function (shader, vbo, ibo, func, use_fb, texid, fb) {
            if (use_fb === void 0) { use_fb = false; }
            if (texid === void 0) { texid = gl.TEXTURE0; }
            if (fb === void 0) { fb = undefined; }
            shader.bind();
            if (use_fb) {
                fb.bindFrameBuffer();
            }
            else {
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            }
            gl.clearColor(0.0, 0.0, 0.0, 1.0);
            gl.clearDepth(1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            vbo.bind(shader);
            ibo.bind();
            func();
            ibo.draw(gl.TRIANGLES);
            if (use_fb) {
                gl.activeTexture(texid);
                gl.bindTexture(gl.TEXTURE_2D, fb.targetTexture);
            }
        };
        return WebGLEnv;
    }());
    EcognitaWeb3D.WebGLEnv = WebGLEnv;
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
/* =========================================================================
 *
 *  FilterViewHub.ts
 *  pkg for filter viewer
 *  v0.1
 *
 * ========================================================================= */
/// <reference path="../lib/EgnType.ts" />
/// <reference path="../lib/EgnWebGL.ts" />
var EcognitaWeb3D;
(function (EcognitaWeb3D) {
    var FilterViewer = /** @class */ (function (_super) {
        __extends(FilterViewer, _super);
        function FilterViewer(cvs) {
            return _super.call(this, cvs) || this;
        }
        FilterViewer.prototype.loadAssets = function () {
            //load demo texture
            this.loadTexture("./image/k0.png", true, gl.CLAMP_TO_BORDER, gl.NEAREST, false);
            this.loadTexture("./image/visual_rgb.png");
            this.loadTexture("./image/lion.png", false);
            this.loadTexture("./image/anim.png", false);
            this.loadTexture("./image/cat.jpg", false);
            this.loadTexture("./image/woman.png", false, gl.CLAMP_TO_EDGE, gl.LINEAR, false);
            this.loadTexture("./image/noise.png", false);
        };
        FilterViewer.prototype.getReqQuery = function () {
            if (window.location.href.split('?').length == 1) {
                return {};
            }
            var queryString = window.location.href.split('?')[1];
            var queryObj = {};
            if (queryString != '') {
                var querys = queryString.split("&");
                for (var i = 0; i < querys.length; i++) {
                    var key = querys[i].split('=')[0];
                    var value = querys[i].split('=')[1];
                    queryObj[key] = value;
                }
            }
            return queryObj;
        };
        FilterViewer.prototype.regisButton = function (btn_data) {
            var _this = this;
            this.btnStatusList = new Utils.HashSet();
            //init button
            btn_data.forEach(function (btn) {
                _this.btnStatusList.set(btn.name, _this.ui_data[btn.name]);
                //register button event
                _this.uiUtil.uiController.get(btn.name).onChange(function (val) {
                    _this.usrSelectChange(btn.name, val, EcognitaWeb3D.RenderPipeLine[btn.pipline], EcognitaWeb3D.Filter[btn.filter], btn.shader);
                });
            });
        };
        FilterViewer.prototype.regisUniforms = function (shader_data) {
            var _this = this;
            shader_data.forEach(function (shader) {
                var uniform_array = new Array();
                var shaderUniforms = shader.uniforms;
                shaderUniforms.forEach(function (uniform) {
                    uniform_array.push(uniform);
                });
                _this.settingUniform(shader.name, uniform_array);
            });
        };
        FilterViewer.prototype.regisUserParam = function (user_config) {
            this.filterMvpMatrix = this.matUtil.identity(this.matUtil.create());
            this.usrParams = user_config.user_params;
            this.usrSelected = user_config.user_selected;
            var default_btn_name = user_config.default_btn;
            var params = this.getReqQuery();
            if (params.p == null || params.f == null || params.s == null || params.b == null) {
                this.usrPipeLine = EcognitaWeb3D.RenderPipeLine[user_config.default_pipline];
                this.usrFilter = EcognitaWeb3D.Filter[user_config.default_filter];
                this.filterShader = this.shaders.get(user_config.default_shader);
            }
            else {
                this.usrPipeLine = EcognitaWeb3D.RenderPipeLine[params.p];
                this.usrFilter = EcognitaWeb3D.Filter[params.f];
                this.filterShader = this.shaders.get(params.s);
                default_btn_name = params.b;
            }
            this.uiData[default_btn_name] = true;
        };
        FilterViewer.prototype.initialize = function (ui_data, shader_data, button_data, user_config) {
            this.uiData = ui_data;
            this.initGlobalVariables();
            this.loadAssets();
            this.loadInternalLibrary(shader_data.shaderList);
            this.regisUniforms(shader_data.shaderList);
            this.regisUserParam(user_config);
            this.loadExtraLibrary(ui_data);
            this.initGlobalMatrix();
            this.regisButton(button_data.buttonList);
            this.initModel();
            this.regisEvent();
            this.settingRenderPipeline();
            this.regisAnimeFunc();
        };
        FilterViewer.prototype.initModel = function () {
            //scene model : torus
            var torusData = new EcognitaMathLib.TorusModel(64, 64, 1.0, 2.0, [1.0, 1.0, 1.0, 1.0], true, false);
            var vbo_torus = new EcognitaMathLib.WebGL_VertexBuffer();
            var ibo_torus = new EcognitaMathLib.WebGL_IndexBuffer();
            this.vbo.push(vbo_torus);
            this.ibo.push(ibo_torus);
            vbo_torus.addAttribute("position", 3, gl.FLOAT, false);
            vbo_torus.addAttribute("normal", 3, gl.FLOAT, false);
            vbo_torus.addAttribute("color", 4, gl.FLOAT, false);
            vbo_torus.init(torusData.data.length / 10);
            vbo_torus.copy(torusData.data);
            ibo_torus.init(torusData.index);
            var position = [
                -1.0, 1.0, 0.0,
                1.0, 1.0, 0.0,
                -1.0, -1.0, 0.0,
                1.0, -1.0, 0.0
            ];
            var boardData = new EcognitaMathLib.BoardModel(position, undefined, false, false, true);
            var vbo_board = new EcognitaMathLib.WebGL_VertexBuffer();
            var ibo_board = new EcognitaMathLib.WebGL_IndexBuffer();
            this.vbo.push(vbo_board);
            this.ibo.push(ibo_board);
            vbo_board.addAttribute("position", 3, gl.FLOAT, false);
            vbo_board.addAttribute("texCoord", 2, gl.FLOAT, false);
            boardData.index = [
                0, 2, 1,
                2, 3, 1
            ];
            vbo_board.init(boardData.data.length / 5);
            vbo_board.copy(boardData.data);
            ibo_board.init(boardData.index);
        };
        FilterViewer.prototype.renderGaussianFilter = function (horizontal, b_gaussian, tex_num) {
            if (tex_num === void 0) { tex_num = 0; }
            var GaussianFilterUniformLoc = this.uniLocations.get("gaussianFilter");
            gl.uniformMatrix4fv(GaussianFilterUniformLoc[0], false, this.filterMvpMatrix);
            gl.uniform1i(GaussianFilterUniformLoc[1], tex_num);
            gl.uniform1fv(GaussianFilterUniformLoc[2], this.usrParams.gaussianWeight);
            gl.uniform1i(GaussianFilterUniformLoc[3], horizontal);
            gl.uniform1f(GaussianFilterUniformLoc[4], this.canvas.height);
            gl.uniform1f(GaussianFilterUniformLoc[5], this.canvas.width);
            gl.uniform1i(GaussianFilterUniformLoc[6], b_gaussian);
        };
        //user config
        FilterViewer.prototype.renderFilter = function () {
            if (this.usrFilter == EcognitaWeb3D.Filter.LAPLACIAN) {
                var LapFilterUniformLoc = this.uniLocations.get("laplacianFilter");
                gl.uniformMatrix4fv(LapFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(LapFilterUniformLoc[1], 0);
                gl.uniform1fv(LapFilterUniformLoc[2], this.usrParams.laplacianCoef);
                gl.uniform1f(LapFilterUniformLoc[3], this.canvas.height);
                gl.uniform1f(LapFilterUniformLoc[4], this.canvas.width);
                gl.uniform1i(LapFilterUniformLoc[5], this.btnStatusList.get("f_LaplacianFilter"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.SOBEL) {
                var SobelFilterUniformLoc = this.uniLocations.get("sobelFilter");
                gl.uniformMatrix4fv(SobelFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(SobelFilterUniformLoc[1], 0);
                gl.uniform1fv(SobelFilterUniformLoc[2], this.usrParams.sobelHorCoef);
                gl.uniform1fv(SobelFilterUniformLoc[3], this.usrParams.sobelVerCoef);
                gl.uniform1f(SobelFilterUniformLoc[4], this.canvas.height);
                gl.uniform1f(SobelFilterUniformLoc[5], this.canvas.width);
                gl.uniform1i(SobelFilterUniformLoc[6], this.btnStatusList.get("f_SobelFilter"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.DoG) {
                var DoGFilterUniformLoc = this.uniLocations.get("DoG");
                gl.uniformMatrix4fv(DoGFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(DoGFilterUniformLoc[1], 0);
                gl.uniform1f(DoGFilterUniformLoc[2], 1.0);
                gl.uniform1f(DoGFilterUniformLoc[3], 1.6);
                gl.uniform1f(DoGFilterUniformLoc[4], 0.99);
                gl.uniform1f(DoGFilterUniformLoc[5], 2.0);
                gl.uniform1f(DoGFilterUniformLoc[6], this.canvas.height);
                gl.uniform1f(DoGFilterUniformLoc[7], this.canvas.width);
                gl.uniform1i(DoGFilterUniformLoc[8], this.btnStatusList.get("f_DoG"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.XDoG) {
                var XDoGFilterUniformLoc = this.uniLocations.get("XDoG");
                gl.uniformMatrix4fv(XDoGFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(XDoGFilterUniformLoc[1], 0);
                gl.uniform1f(XDoGFilterUniformLoc[2], 1.4);
                gl.uniform1f(XDoGFilterUniformLoc[3], 1.6);
                gl.uniform1f(XDoGFilterUniformLoc[4], 21.7);
                gl.uniform1f(XDoGFilterUniformLoc[5], 79.5);
                gl.uniform1f(XDoGFilterUniformLoc[6], 0.017);
                gl.uniform1f(XDoGFilterUniformLoc[7], this.canvas.height);
                gl.uniform1f(XDoGFilterUniformLoc[8], this.canvas.width);
                gl.uniform1i(XDoGFilterUniformLoc[9], this.btnStatusList.get("f_XDoG"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.KUWAHARA) {
                var KuwaharaFilterUniformLoc = this.uniLocations.get("kuwaharaFilter");
                gl.uniformMatrix4fv(KuwaharaFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(KuwaharaFilterUniformLoc[1], 0);
                gl.uniform1f(KuwaharaFilterUniformLoc[2], this.canvas.height);
                gl.uniform1f(KuwaharaFilterUniformLoc[3], this.canvas.width);
                gl.uniform1i(KuwaharaFilterUniformLoc[4], this.btnStatusList.get("f_KuwaharaFilter"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.GKUWAHARA) {
                var GKuwaharaFilterUniformLoc = this.uniLocations.get("gkuwaharaFilter");
                gl.uniformMatrix4fv(GKuwaharaFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(GKuwaharaFilterUniformLoc[1], 0);
                gl.uniform1fv(GKuwaharaFilterUniformLoc[2], this.usrParams.gkweight);
                gl.uniform1f(GKuwaharaFilterUniformLoc[3], this.canvas.height);
                gl.uniform1f(GKuwaharaFilterUniformLoc[4], this.canvas.width);
                gl.uniform1i(GKuwaharaFilterUniformLoc[5], this.btnStatusList.get("f_GeneralizedKuwaharaFilter"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.ANISTROPIC) {
                var AnisotropicFilterUniformLoc = this.uniLocations.get("Anisotropic");
                gl.uniformMatrix4fv(AnisotropicFilterUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(AnisotropicFilterUniformLoc[1], 0);
                gl.uniform1i(AnisotropicFilterUniformLoc[2], 1);
                gl.uniform1i(AnisotropicFilterUniformLoc[3], 2);
                gl.uniform1f(AnisotropicFilterUniformLoc[4], this.canvas.height);
                gl.uniform1f(AnisotropicFilterUniformLoc[5], this.canvas.width);
                gl.uniform1i(AnisotropicFilterUniformLoc[6], this.btnStatusList.get("f_VisualAnisotropic"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.AKUWAHARA) {
                var AKFUniformLoc = this.uniLocations.get("AKF");
                gl.uniformMatrix4fv(AKFUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(AKFUniformLoc[1], 0);
                gl.uniform1i(AKFUniformLoc[2], 1);
                gl.uniform1i(AKFUniformLoc[3], 2);
                gl.uniform1f(AKFUniformLoc[4], 6.0);
                gl.uniform1f(AKFUniformLoc[5], 8.0);
                gl.uniform1f(AKFUniformLoc[6], 1.0);
                gl.uniform1f(AKFUniformLoc[7], this.canvas.height);
                gl.uniform1f(AKFUniformLoc[8], this.canvas.width);
                gl.uniform1i(AKFUniformLoc[9], this.btnStatusList.get("f_AnisotropicKuwahara"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.LIC || this.usrFilter == EcognitaWeb3D.Filter.NOISELIC) {
                var LICUniformLoc = this.uniLocations.get("LIC");
                gl.uniformMatrix4fv(LICUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(LICUniformLoc[1], 0);
                gl.uniform1i(LICUniformLoc[2], 1);
                gl.uniform1f(LICUniformLoc[3], 3.0);
                gl.uniform1f(LICUniformLoc[4], this.canvas.height);
                gl.uniform1f(LICUniformLoc[5], this.canvas.width);
                if (this.usrFilter == EcognitaWeb3D.Filter.LIC) {
                    gl.uniform1i(LICUniformLoc[6], this.btnStatusList.get("f_LIC"));
                }
                else if (this.usrFilter == EcognitaWeb3D.Filter.NOISELIC) {
                    gl.uniform1i(LICUniformLoc[6], this.btnStatusList.get("f_NoiseLIC"));
                }
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.FDoG) {
                var FDoGUniformLoc = this.uniLocations.get("FDoG");
                gl.uniformMatrix4fv(FDoGUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(FDoGUniformLoc[1], 0);
                gl.uniform1i(FDoGUniformLoc[2], 1);
                gl.uniform1f(FDoGUniformLoc[3], 3.0);
                gl.uniform1f(FDoGUniformLoc[4], 2.0);
                gl.uniform1f(FDoGUniformLoc[5], this.canvas.height);
                gl.uniform1f(FDoGUniformLoc[6], this.canvas.width);
                gl.uniform1i(FDoGUniformLoc[7], this.btnStatusList.get("f_FDoG"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.FXDoG) {
                var FXDoGUniformLoc = this.uniLocations.get("FXDoG");
                gl.uniformMatrix4fv(FXDoGUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(FXDoGUniformLoc[1], 0);
                gl.uniform1i(FXDoGUniformLoc[2], 1);
                gl.uniform1f(FXDoGUniformLoc[3], 4.4);
                gl.uniform1f(FXDoGUniformLoc[4], 0.017);
                gl.uniform1f(FXDoGUniformLoc[5], 79.5);
                gl.uniform1f(FXDoGUniformLoc[6], this.canvas.height);
                gl.uniform1f(FXDoGUniformLoc[7], this.canvas.width);
                gl.uniform1i(FXDoGUniformLoc[8], this.btnStatusList.get("f_FXDoG"));
            }
            else if (this.usrFilter == EcognitaWeb3D.Filter.ABSTRACTION) {
                var ABSUniformLoc = this.uniLocations.get("Abstraction");
                gl.uniformMatrix4fv(ABSUniformLoc[0], false, this.filterMvpMatrix);
                gl.uniform1i(ABSUniformLoc[1], 1);
                gl.uniform1i(ABSUniformLoc[2], 3);
                gl.uniform1i(ABSUniformLoc[3], 4);
                gl.uniform3fv(ABSUniformLoc[4], [0.0, 0.0, 0.0]);
                gl.uniform1f(ABSUniformLoc[5], this.canvas.height);
                gl.uniform1f(ABSUniformLoc[6], this.canvas.width);
                gl.uniform1i(ABSUniformLoc[7], this.btnStatusList.get("f_Abstraction"));
            }
        };
        FilterViewer.prototype.settingUniform = function (shaderName, uniformIndexArray) {
            var uniLocArray = this.uniLocations.get(shaderName);
            var shader = this.shaders.get(shaderName);
            uniformIndexArray.forEach(function (uniName) {
                uniLocArray.push(shader.uniformIndex(uniName));
            });
        };
        FilterViewer.prototype.settingRenderPipeline = function () {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
            gl.enable(gl.CULL_FACE);
        };
        FilterViewer.prototype.usrSelectChange = function (btnName, val, pipeline, filter, filter_name) {
            this.btnStatusList.set(btnName, val);
            if (val) {
                this.usrPipeLine = pipeline;
                this.usrFilter = filter;
                this.filterShader = this.shaders.get(filter_name);
                for (var key in this.ui_data) {
                    if (key.includes('_')) {
                        var f_name = key.split("_");
                        if (f_name[0] == "f" && key != btnName) {
                            //un select other btn
                            this.btnStatusList.set(key, !val);
                            this.ui_data[key] = !val;
                        }
                    }
                }
            }
        };
        FilterViewer.prototype.regisEvent = function () {
            var _this = this;
            //show tool tip for canvas
            $('[data-toggle="tooltip"]').tooltip();
            //select event
            $("select").imagepicker({
                hide_select: true,
                show_label: false,
                selected: function () {
                    _this.usrSelected = $("select").val();
                }
            });
            //drop image event 
            this.canvas.addEventListener('dragover', function (event) {
                event.preventDefault();
            });
            this.canvas.addEventListener('drop', function (event) {
                event.preventDefault();
                var fileData = event.dataTransfer.files[0];
                if (!fileData.type.match('image.*')) {
                    alert('you should upload a image!');
                    return;
                }
                var reader = new FileReader();
                reader.onload = (function () {
                    _this.loadTexture(reader.result, false, gl.CLAMP_TO_EDGE, gl.LINEAR, false);
                    _this.usrSelected = reader.result;
                });
                reader.readAsDataURL(fileData);
            });
            //touch event
            var lastPosX = 0;
            var lastPosY = 0;
            var isDragging = false;
            var hammer = this.extHammer;
            hammer.on_pan = function (ev) {
                var elem = ev.target;
                if (!isDragging) {
                    isDragging = true;
                    lastPosX = elem.offsetLeft;
                    lastPosY = elem.offsetTop;
                }
                var posX = ev.center.x - lastPosX;
                var posY = ev.center.y - lastPosY;
                var cw = _this.canvas.width;
                var ch = _this.canvas.height;
                var wh = 1 / Math.sqrt(cw * cw + ch * ch);
                var x = posX - cw * 0.5;
                var y = posY - ch * 0.5;
                var sq = Math.sqrt(x * x + y * y);
                var r = sq * 2.0 * Math.PI * wh;
                if (sq != 1) {
                    sq = 1 / sq;
                    x *= sq;
                    y *= sq;
                }
                _this.usrQuaternion = _this.quatUtil.rotate(r, [y, x, 0.0]);
                if (ev.isFinal) {
                    isDragging = false;
                }
            };
            hammer.enablePan();
        };
        FilterViewer.prototype.regisFrameBuffer = function (num) {
            var fArray = Array(num);
            for (var i = 0; i < num; i++) {
                this.settingFrameBuffer("frameBuffer" + i);
                var fb = this.framebuffers.get("frameBuffer" + i);
                fArray[i] = fb;
            }
            return fArray;
        };
        FilterViewer.prototype.regisAnimeFunc = function () {
            var _this = this;
            //fundmental setting and variables
            var cnt = 0;
            var cnt1 = 0;
            var lightDirection = [-0.577, 0.577, 0.577];
            var m = this.matUtil;
            var q = this.quatUtil;
            this.usrQuaternion = q.identity(q.create());
            //init scene and model
            var sceneShader = this.shaders.get("filterScene");
            var sceneUniformLoc = this.uniLocations.get("filterScene");
            var vbo_torus = this.vbo[0];
            var ibo_torus = this.ibo[0];
            var vbo_board = this.vbo[1];
            var ibo_board = this.ibo[1];
            var mMatrix = this.MATRIX.get("mMatrix");
            var vMatrix = this.MATRIX.get("vMatrix");
            var pMatrix = this.MATRIX.get("pMatrix");
            var vpMatrix = this.MATRIX.get("vpMatrix");
            var mvpMatrix = this.MATRIX.get("mvpMatrix");
            var invMatrix = this.MATRIX.get("invMatrix");
            //------------------------------------user config
            var specCptShader = this.shaders.get("specCpt");
            var uniLocation_spec = this.uniLocations.get("specCpt");
            var synthShader = this.shaders.get("synth");
            var uniLocation_synth = this.uniLocations.get("synth");
            //get luminance
            var luminanceShader = this.shaders.get("luminance");
            var uniLocation_luminance = this.uniLocations.get("luminance");
            var TFShader = this.shaders.get("TF");
            var uniLocation_TF = this.uniLocations.get("TF");
            var ETFShader = this.shaders.get("ETF");
            var uniLocation_ETF = this.uniLocations.get("ETF");
            //DoG XDoG
            var PFDoGShader = this.shaders.get("P_FDoG");
            var uniLocation_PFDoG = this.uniLocations.get("P_FDoG");
            var PFXDoGShader = this.shaders.get("P_FXDoG");
            var uniLocation_PFXDoG = this.uniLocations.get("P_FXDoG");
            var FXDoGShader = this.shaders.get("FXDoG");
            var uniLocation_FXDoG = this.uniLocations.get("FXDoG");
            //anisotropic
            var SSTShader = this.shaders.get("SST");
            var uniLocation_SST = this.uniLocations.get("SST");
            var GAUShader = this.shaders.get("Gaussian_K");
            var uniLocation_GAU = this.uniLocations.get("Gaussian_K");
            var TFMShader = this.shaders.get("TFM");
            var uniLocation_TFM = this.uniLocations.get("TFM");
            var AKFShader = this.shaders.get("AKF");
            var AKFUniformLoc = this.uniLocations.get("AKF");
            //-----------------------------------------------
            //get framebuffer
            var fb = this.regisFrameBuffer(5);
            var loop = function () {
                //--------------------------------------animation global variables
                _this.stats.begin();
                cnt++;
                if (cnt % 2 == 0) {
                    cnt1++;
                }
                var rad = (cnt % 360) * Math.PI / 180;
                var eyePosition = new Array();
                var camUpDirection = new Array();
                eyePosition = q.ToV3([0.0, 20.0, 0.0], _this.usrQuaternion);
                camUpDirection = q.ToV3([0.0, 0.0, -1.0], _this.usrQuaternion);
                //camera setting
                vMatrix = m.viewMatrix(eyePosition, [0, 0, 0], camUpDirection);
                pMatrix = m.perspectiveMatrix(90, _this.canvas.width / _this.canvas.height, 0.1, 100);
                vpMatrix = m.multiply(pMatrix, vMatrix);
                // orth matrix
                vMatrix = m.viewMatrix([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
                pMatrix = m.orthoMatrix(-1.0, 1.0, 1.0, -1.0, 0.1, 1);
                _this.filterMvpMatrix = m.multiply(pMatrix, vMatrix);
                //--------------------------------------animation global variables
                //rendering parts----------------------------------------------------------------------------------
                var inTex = _this.Texture.get(_this.usrSelected);
                if (_this.usrPipeLine == EcognitaWeb3D.RenderPipeLine.CONVOLUTION_FILTER) {
                    //---------------------using framebuffer1 to render scene and save result to texture0
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene);
                    }
                    _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () { _this.renderFilter(); });
                }
                else if (_this.usrPipeLine == EcognitaWeb3D.RenderPipeLine.BLOOM_EFFECT) {
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        //get texture
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        //render scene specular parts
                        // this.renderSceneByFrameBuffer(fb[0],RenderSimpleSceneSpecularParts);
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene);
                    }
                    //get brightness parts
                    _this.renderBoardByFrameBuffer(luminanceShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_luminance[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_luminance[1], 0);
                        gl.uniform1f(uniLocation_luminance[2], 0.5);
                    }, true, gl.TEXTURE1, fb[1]);
                    var sample_count = 9;
                    for (var _i = 0; _i < sample_count; _i++) {
                        //horizontal blur, save to frame2
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            _this.renderGaussianFilter(true, _this.btnStatusList.get("f_BloomEffect"), 1);
                        }, true, gl.TEXTURE0, fb[0]);
                        //vertical blur,save to frame1 and render to texture1
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            _this.renderGaussianFilter(false, _this.btnStatusList.get("f_BloomEffect"));
                        }, true, gl.TEXTURE1, fb[1]);
                    }
                    //render scene, save to texture0
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            RenderSimpleScene();
                        }, true, gl.TEXTURE0, fb[0]);
                    }
                    //synthsis texture0 and texture1
                    _this.renderBoardByFrameBuffer(synthShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_synth[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_synth[1], 0);
                        gl.uniform1i(uniLocation_synth[2], 1);
                        gl.uniform1i(uniLocation_synth[3], _this.btnStatusList.get("f_BloomEffect"));
                    });
                }
                else if (_this.usrPipeLine == EcognitaWeb3D.RenderPipeLine.CONVOLUTION_TWICE) {
                    //---------------------using framebuffer1 to render scene and save result to texture0
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene);
                    }
                    //horizontal blur, save to frame2
                    if (_this.btnStatusList.get("f_GaussianFilter")) {
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            _this.renderGaussianFilter(true, _this.btnStatusList.get("f_GaussianFilter"));
                        }, true, gl.TEXTURE0, fb[1]);
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            _this.renderGaussianFilter(false, _this.btnStatusList.get("f_GaussianFilter"));
                        });
                    }
                    else {
                        _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                            _this.renderGaussianFilter(false, _this.btnStatusList.get("f_GaussianFilter"));
                        });
                    }
                }
                else if (_this.usrPipeLine == EcognitaWeb3D.RenderPipeLine.ANISTROPIC) {
                    if (_this.usrFilter == EcognitaWeb3D.Filter.ANISTROPIC) {
                        var visTex = _this.Texture.get("./image/visual_rgb.png");
                        if (visTex != undefined) {
                            gl.activeTexture(gl.TEXTURE2);
                            visTex.bind(visTex.texture);
                        }
                    }
                    else if (_this.usrFilter == EcognitaWeb3D.Filter.AKUWAHARA) {
                        //save k0 texture to tex2
                        var k0Tex = _this.Texture.get("./image/k0.png");
                        if (k0Tex != undefined) {
                            gl.activeTexture(gl.TEXTURE2);
                            k0Tex.bind(k0Tex.texture);
                        }
                    }
                    //render SRC
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene);
                    }
                    //render SST
                    _this.renderBoardByFrameBuffer(SSTShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_SST[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_SST[1], 0);
                        gl.uniform1f(uniLocation_SST[2], _this.canvas.height);
                        gl.uniform1f(uniLocation_SST[3], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[1]);
                    //render Gaussian
                    _this.renderBoardByFrameBuffer(GAUShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_GAU[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_GAU[1], 0);
                        gl.uniform1f(uniLocation_GAU[2], 2.0);
                        gl.uniform1f(uniLocation_GAU[3], _this.canvas.height);
                        gl.uniform1f(uniLocation_GAU[4], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[0]);
                    //render TFM
                    _this.renderBoardByFrameBuffer(TFMShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_TFM[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_TFM[1], 0);
                        gl.uniform1f(uniLocation_TFM[2], _this.canvas.height);
                        gl.uniform1f(uniLocation_TFM[3], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[1]);
                    if (_this.usrFilter == EcognitaWeb3D.Filter.NOISELIC) {
                        gl.activeTexture(gl.TEXTURE1);
                        var noiseTex = _this.Texture.get("./image/noise.png");
                        if (noiseTex != undefined) {
                            noiseTex.bind(noiseTex.texture);
                        }
                    }
                    else {
                        if (inTex != undefined && _this.ui_data.useTexture) {
                            gl.activeTexture(gl.TEXTURE1);
                            inTex.bind(inTex.texture);
                        }
                        else {
                            _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene, gl.TEXTURE1);
                        }
                    }
                    //FDoG pre-calculation
                    if (_this.usrFilter == EcognitaWeb3D.Filter.FDoG) {
                        _this.renderBoardByFrameBuffer(PFDoGShader, vbo_board, ibo_board, function () {
                            gl.uniformMatrix4fv(uniLocation_PFDoG[0], false, _this.filterMvpMatrix);
                            gl.uniform1i(uniLocation_PFDoG[1], 0);
                            gl.uniform1i(uniLocation_PFDoG[2], 1);
                            gl.uniform1f(uniLocation_PFDoG[3], 1.0);
                            gl.uniform1f(uniLocation_PFDoG[4], 1.6);
                            gl.uniform1f(uniLocation_PFDoG[5], 0.99);
                            gl.uniform1f(uniLocation_PFDoG[6], _this.canvas.height);
                            gl.uniform1f(uniLocation_PFDoG[7], _this.canvas.width);
                            gl.uniform1i(uniLocation_PFDoG[8], _this.btnStatusList.get("f_FDoG"));
                        }, true, gl.TEXTURE1, fb[2]);
                    }
                    else if (_this.usrFilter == EcognitaWeb3D.Filter.FXDoG) {
                        _this.renderBoardByFrameBuffer(PFXDoGShader, vbo_board, ibo_board, function () {
                            gl.uniformMatrix4fv(uniLocation_PFXDoG[0], false, _this.filterMvpMatrix);
                            gl.uniform1i(uniLocation_PFXDoG[1], 0);
                            gl.uniform1i(uniLocation_PFXDoG[2], 1);
                            gl.uniform1f(uniLocation_PFXDoG[3], 1.4);
                            gl.uniform1f(uniLocation_PFXDoG[4], 1.6);
                            gl.uniform1f(uniLocation_PFXDoG[5], 21.7);
                            gl.uniform1f(uniLocation_PFXDoG[6], _this.canvas.height);
                            gl.uniform1f(uniLocation_PFXDoG[7], _this.canvas.width);
                            gl.uniform1i(uniLocation_PFXDoG[8], _this.btnStatusList.get("f_FXDoG"));
                        }, true, gl.TEXTURE1, fb[2]);
                    }
                    //render Filter
                    _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                        _this.renderFilter();
                    });
                }
                else if (_this.usrPipeLine == EcognitaWeb3D.RenderPipeLine.ABSTRACTION) {
                    //get k0 texture
                    var k0Tex = _this.Texture.get("./image/k0.png");
                    if (k0Tex != undefined) {
                        gl.activeTexture(gl.TEXTURE2);
                        k0Tex.bind(k0Tex.texture);
                    }
                    //tex0/f2: render TFM
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene);
                    }
                    //render SST
                    _this.renderBoardByFrameBuffer(SSTShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_SST[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_SST[1], 0);
                        gl.uniform1f(uniLocation_SST[2], _this.canvas.height);
                        gl.uniform1f(uniLocation_SST[3], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[1]);
                    //render Gaussian
                    _this.renderBoardByFrameBuffer(GAUShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_GAU[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_GAU[1], 0);
                        gl.uniform1f(uniLocation_GAU[2], 2.0);
                        gl.uniform1f(uniLocation_GAU[3], _this.canvas.height);
                        gl.uniform1f(uniLocation_GAU[4], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[0]);
                    //render TFM
                    _this.renderBoardByFrameBuffer(TFMShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_TFM[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_TFM[1], 0);
                        gl.uniform1f(uniLocation_TFM[2], _this.canvas.height);
                        gl.uniform1f(uniLocation_TFM[3], _this.canvas.width);
                    }, true, gl.TEXTURE0, fb[1]);
                    //tex1/f1:src
                    if (inTex != undefined && _this.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE1);
                        inTex.bind(inTex.texture);
                    }
                    else {
                        _this.renderSceneByFrameBuffer(fb[0], RenderSimpleScene, gl.TEXTURE1);
                    }
                    //tex3/f3:akf
                    _this.renderBoardByFrameBuffer(AKFShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(AKFUniformLoc[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(AKFUniformLoc[1], 0);
                        gl.uniform1i(AKFUniformLoc[2], 1);
                        gl.uniform1i(AKFUniformLoc[3], 2);
                        gl.uniform1f(AKFUniformLoc[4], 6.0);
                        gl.uniform1f(AKFUniformLoc[5], 8.0);
                        gl.uniform1f(AKFUniformLoc[6], 1.0);
                        gl.uniform1f(AKFUniformLoc[7], _this.canvas.height);
                        gl.uniform1f(AKFUniformLoc[8], _this.canvas.width);
                        gl.uniform1i(AKFUniformLoc[9], _this.btnStatusList.get("f_Abstraction"));
                    }, true, gl.TEXTURE3, fb[2]);
                    //tex4/f4:fxdog
                    _this.renderBoardByFrameBuffer(PFXDoGShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_PFXDoG[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_PFXDoG[1], 0);
                        gl.uniform1i(uniLocation_PFXDoG[2], 1);
                        gl.uniform1f(uniLocation_PFXDoG[3], 1.4);
                        gl.uniform1f(uniLocation_PFXDoG[4], 1.6);
                        gl.uniform1f(uniLocation_PFXDoG[5], 21.7);
                        gl.uniform1f(uniLocation_PFXDoG[6], _this.canvas.height);
                        gl.uniform1f(uniLocation_PFXDoG[7], _this.canvas.width);
                        gl.uniform1i(uniLocation_PFXDoG[8], _this.btnStatusList.get("f_Abstraction"));
                    }, true, gl.TEXTURE4, fb[3]);
                    _this.renderBoardByFrameBuffer(FXDoGShader, vbo_board, ibo_board, function () {
                        gl.uniformMatrix4fv(uniLocation_FXDoG[0], false, _this.filterMvpMatrix);
                        gl.uniform1i(uniLocation_FXDoG[1], 0);
                        gl.uniform1i(uniLocation_FXDoG[2], 4);
                        gl.uniform1f(uniLocation_FXDoG[3], 4.4);
                        gl.uniform1f(uniLocation_FXDoG[4], 0.017);
                        gl.uniform1f(uniLocation_FXDoG[5], 79.5);
                        gl.uniform1f(uniLocation_FXDoG[6], _this.canvas.height);
                        gl.uniform1f(uniLocation_FXDoG[7], _this.canvas.width);
                        gl.uniform1i(uniLocation_FXDoG[8], _this.btnStatusList.get("f_Abstraction"));
                    }, true, gl.TEXTURE4, fb[4]);
                    //render Abstratcion Filter
                    _this.renderBoardByFrameBuffer(_this.filterShader, vbo_board, ibo_board, function () {
                        _this.renderFilter();
                    });
                }
                //rendering parts----------------------------------------------------------------------------------
                //--------------------------------------animation global variables
                gl.flush();
                _this.stats.end();
                requestAnimationFrame(loop);
                function RenderSimpleScene() {
                    sceneShader.bind();
                    var hsv = EcognitaMathLib.HSV2RGB(cnt1 % 360, 1, 1, 1);
                    gl.clearColor(hsv[0], hsv[1], hsv[2], hsv[3]);
                    gl.clearDepth(1.0);
                    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    vbo_torus.bind(sceneShader);
                    ibo_torus.bind();
                    for (var i = 0; i < 9; i++) {
                        var amb = EcognitaMathLib.HSV2RGB(i * 40, 1, 1, 1);
                        mMatrix = m.identity(mMatrix);
                        mMatrix = m.rotate(mMatrix, i * 2 * Math.PI / 9, [0, 1, 0]);
                        mMatrix = m.translate(mMatrix, [0.0, 0.0, 10.0]);
                        mMatrix = m.rotate(mMatrix, rad, [1, 1, 0]);
                        mvpMatrix = m.multiply(vpMatrix, mMatrix);
                        invMatrix = m.inverse(mMatrix);
                        gl.uniformMatrix4fv(sceneUniformLoc[0], false, mvpMatrix);
                        gl.uniformMatrix4fv(sceneUniformLoc[1], false, invMatrix);
                        gl.uniform3fv(sceneUniformLoc[2], lightDirection);
                        gl.uniform3fv(sceneUniformLoc[3], eyePosition);
                        gl.uniform4fv(sceneUniformLoc[4], amb);
                        ibo_torus.draw(gl.TRIANGLES);
                    }
                }
                function RenderSimpleSceneSpecularParts() {
                    specCptShader.bind();
                    gl.clearColor(0.0, 0.0, 0.0, 1.0);
                    gl.clearDepth(1.0);
                    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    vbo_torus.bind(specCptShader);
                    ibo_torus.bind();
                    for (var i = 0; i < 9; i++) {
                        mMatrix = m.identity(mMatrix);
                        mMatrix = m.rotate(mMatrix, i * 2 * Math.PI / 9, [0, 1, 0]);
                        mMatrix = m.translate(mMatrix, [0.0, 0.0, 10.0]);
                        mMatrix = m.rotate(mMatrix, rad, [1, 1, 0]);
                        mvpMatrix = m.multiply(vpMatrix, mMatrix);
                        invMatrix = m.inverse(mMatrix);
                        gl.uniformMatrix4fv(uniLocation_spec[0], false, mvpMatrix);
                        gl.uniformMatrix4fv(uniLocation_spec[1], false, invMatrix);
                        gl.uniform3fv(uniLocation_spec[2], lightDirection);
                        gl.uniform3fv(uniLocation_spec[3], eyePosition);
                        ibo_torus.draw(gl.TRIANGLES);
                    }
                }
                //--------------------------------------animation global variables
            };
            loop();
        };
        return FilterViewer;
    }(EcognitaWeb3D.WebGLEnv));
    EcognitaWeb3D.FilterViewer = FilterViewer;
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
/// <reference path="../ts_scripts/package/pkg_FilterViewHub.ts" />
var viewer = document.getElementById("canvas_viewer");
viewer.width = 1920;
viewer.height = 1080;
//load data
$.getJSON("./config/ui.json", function (ui_data) {
    //console.log( "loaded UI data");
    $.getJSON("./config/shader.json", function (shader_data) {
        //console.log( "loaded shader data");
        $.getJSON("./config/button.json", function (button_data) {
            //console.log( "loaded button data");
            $.getJSON("./config/user_config.json", function (user_config) {
                //console.log( "loaded user config data");
                var filterViewer = new EcognitaWeb3D.FilterViewer(viewer);
                filterViewer.initialize(ui_data, shader_data, button_data, user_config);
            });
        });
    });
});
