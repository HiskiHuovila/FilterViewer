var __extends =
    (this && this.__extends) ||
    (function () {
        var a = function (e, c) {
            a =
                Object.setPrototypeOf ||
                ({ __proto__: [] } instanceof Array &&
                    function (g, f) {
                        g.__proto__ = f;
                    }) ||
                function (h, f) {
                    for (var g in f) {
                        if (f.hasOwnProperty(g)) {
                            h[g] = f[g];
                        }
                    }
                };
            return a(e, c);
        };
        return function (f, c) {
            a(f, c);
            function e() {
                this.constructor = f;
            }
            f.prototype = c === null ? Object.create(c) : ((e.prototype = c.prototype), new e());
        };
    })();
var EcognitaWeb3D;
(function (c) {
    var a;
    (function (d) {
        d[(d.LAPLACIAN = 0)] = "LAPLACIAN";
        d[(d.SOBEL = 1)] = "SOBEL";
        d[(d.GAUSSIAN = 2)] = "GAUSSIAN";
        d[(d.KUWAHARA = 3)] = "KUWAHARA";
        d[(d.GKUWAHARA = 4)] = "GKUWAHARA";
        d[(d.AKUWAHARA = 5)] = "AKUWAHARA";
        d[(d.ANISTROPIC = 6)] = "ANISTROPIC";
        d[(d.LIC = 7)] = "LIC";
        d[(d.NOISELIC = 8)] = "NOISELIC";
        d[(d.DoG = 9)] = "DoG";
        d[(d.XDoG = 10)] = "XDoG";
        d[(d.FDoG = 11)] = "FDoG";
        d[(d.FXDoG = 12)] = "FXDoG";
        d[(d.ABSTRACTION = 13)] = "ABSTRACTION";
    })((a = c.Filter || (c.Filter = {})));
    var b;
    (function (d) {
        d[(d.CONVOLUTION_FILTER = 0)] = "CONVOLUTION_FILTER";
        d[(d.ANISTROPIC = 1)] = "ANISTROPIC";
        d[(d.BLOOM_EFFECT = 2)] = "BLOOM_EFFECT";
        d[(d.CONVOLUTION_TWICE = 3)] = "CONVOLUTION_TWICE";
        d[(d.ABSTRACTION = 4)] = "ABSTRACTION";
    })((b = c.RenderPipeLine || (c.RenderPipeLine = {})));
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
var EcognitaMathLib;
(function (a) {
    function b(d) {
        var c = new Image();
        c.src = d;
        return c;
    }
    a.imread = b;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var EcognitaMathLib;
(function (b) {
    function a(q, z, y, x) {
        if (z > 1 || y > 1 || x > 1) {
            return;
        }
        var d = q % 360;
        var p = Math.floor(d / 60);
        var u = d / 60 - p;
        var j = y * (1 - z);
        var e = y * (1 - z * u);
        var o = y * (1 - z * (1 - u));
        var l = new Array();
        if (!(z > 0) && !(z < 0)) {
            l.push(y, y, y, x);
        } else {
            var c = new Array(y, e, j, j, o, y);
            var t = new Array(o, y, y, e, j, j);
            var w = new Array(j, j, o, y, y, e);
            l.push(c[p], t[p], w[p], x);
        }
        return l;
    }
    b.HSV2RGB = a;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var EcognitaMathLib;
(function (b) {
    var a = (function () {
        function c(d) {
            this.hm = new Hammer(d);
            this.on_pan = undefined;
        }
        c.prototype.enablePan = function () {
            if (this.on_pan == undefined) {
                console.log("please setting the PAN function!");
                return;
            }
            this.hm.add(new Hammer.Pan({ direction: Hammer.DIRECTION_ALL, threshold: 0 }));
            this.hm.on("pan", this.on_pan);
        };
        return c;
    })();
    b.Hammer_Utils = a;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var EcognitaMathLib;
(function (a) {
    var b = (function () {
        function c() {
            this.inverse = function (D) {
                var ab = this.create();
                var ag = D[0],
                    af = D[1],
                    ae = D[2],
                    ad = D[3],
                    ac = D[4],
                    aa = D[5],
                    Z = D[6],
                    Y = D[7],
                    X = D[8],
                    W = D[9],
                    V = D[10],
                    U = D[11],
                    T = D[12],
                    S = D[13],
                    R = D[14],
                    Q = D[15],
                    O = ag * aa - af * ac,
                    M = ag * Z - ae * ac,
                    L = ag * Y - ad * ac,
                    K = af * Z - ae * aa,
                    J = af * Y - ad * aa,
                    I = ae * Y - ad * Z,
                    H = X * S - W * T,
                    G = X * R - V * T,
                    F = X * Q - U * T,
                    E = W * R - V * S,
                    P = W * Q - U * S,
                    N = V * Q - U * R,
                    C = 1 / (O * N - M * P + L * E + K * F - J * G + I * H);
                ab[0] = (aa * N - Z * P + Y * E) * C;
                ab[1] = (-af * N + ae * P - ad * E) * C;
                ab[2] = (S * I - R * J + Q * K) * C;
                ab[3] = (-W * I + V * J - U * K) * C;
                ab[4] = (-ac * N + Z * F - Y * G) * C;
                ab[5] = (ag * N - ae * F + ad * G) * C;
                ab[6] = (-T * I + R * L - Q * M) * C;
                ab[7] = (X * I - V * L + U * M) * C;
                ab[8] = (ac * P - aa * F + Y * H) * C;
                ab[9] = (-ag * P + af * F - ad * H) * C;
                ab[10] = (T * J - S * L + Q * O) * C;
                ab[11] = (-X * J + W * L - U * O) * C;
                ab[12] = (-ac * E + aa * G - Z * H) * C;
                ab[13] = (ag * E - af * G + ae * H) * C;
                ab[14] = (-T * K + S * M - R * O) * C;
                ab[15] = (X * K - W * M + V * O) * C;
                return ab;
            };
        }
        c.prototype.create = function () {
            return new Float32Array(16);
        };
        c.prototype.identity = function (d) {
            d[0] = 1;
            d[1] = 0;
            d[2] = 0;
            d[3] = 0;
            d[4] = 0;
            d[5] = 1;
            d[6] = 0;
            d[7] = 0;
            d[8] = 0;
            d[9] = 0;
            d[10] = 1;
            d[11] = 0;
            d[12] = 0;
            d[13] = 0;
            d[14] = 0;
            d[15] = 1;
            return d;
        };
        c.prototype.multiply = function (u, s) {
            var aj = this.create();
            var ao = u[0],
                an = u[1],
                am = u[2],
                al = u[3],
                ak = u[4],
                ai = u[5],
                ah = u[6],
                ag = u[7],
                af = u[8],
                ae = u[9],
                ad = u[10],
                ac = u[11],
                ab = u[12],
                aa = u[13],
                Z = u[14],
                Y = u[15],
                X = s[0],
                W = s[1],
                V = s[2],
                U = s[3],
                T = s[4],
                S = s[5],
                R = s[6],
                Q = s[7],
                z = s[8],
                y = s[9],
                x = s[10],
                w = s[11],
                v = s[12],
                t = s[13],
                r = s[14],
                q = s[15];
            aj[0] = X * ao + W * ak + V * af + U * ab;
            aj[1] = X * an + W * ai + V * ae + U * aa;
            aj[2] = X * am + W * ah + V * ad + U * Z;
            aj[3] = X * al + W * ag + V * ac + U * Y;
            aj[4] = T * ao + S * ak + R * af + Q * ab;
            aj[5] = T * an + S * ai + R * ae + Q * aa;
            aj[6] = T * am + S * ah + R * ad + Q * Z;
            aj[7] = T * al + S * ag + R * ac + Q * Y;
            aj[8] = z * ao + y * ak + x * af + w * ab;
            aj[9] = z * an + y * ai + x * ae + w * aa;
            aj[10] = z * am + y * ah + x * ad + w * Z;
            aj[11] = z * al + y * ag + x * ac + w * Y;
            aj[12] = v * ao + t * ak + r * af + q * ab;
            aj[13] = v * an + t * ai + r * ae + q * aa;
            aj[14] = v * am + t * ah + r * ad + q * Z;
            aj[15] = v * al + t * ag + r * ac + q * Y;
            return aj;
        };
        c.prototype.scale = function (f, d) {
            var e = this.create();
            if (d.length != 3) {
                return undefined;
            }
            e[0] = f[0] * d[0];
            e[1] = f[1] * d[0];
            e[2] = f[2] * d[0];
            e[3] = f[3] * d[0];
            e[4] = f[4] * d[1];
            e[5] = f[5] * d[1];
            e[6] = f[6] * d[1];
            e[7] = f[7] * d[1];
            e[8] = f[8] * d[2];
            e[9] = f[9] * d[2];
            e[10] = f[10] * d[2];
            e[11] = f[11] * d[2];
            e[12] = f[12];
            e[13] = f[13];
            e[14] = f[14];
            e[15] = f[15];
            return e;
        };
        c.prototype.translate = function (f, d) {
            var e = this.create();
            if (d.length != 3) {
                return undefined;
            }
            e[0] = f[0];
            e[1] = f[1];
            e[2] = f[2];
            e[3] = f[3];
            e[4] = f[4];
            e[5] = f[5];
            e[6] = f[6];
            e[7] = f[7];
            e[8] = f[8];
            e[9] = f[9];
            e[10] = f[10];
            e[11] = f[11];
            e[12] = f[0] * d[0] + f[4] * d[1] + f[8] * d[2] + f[12];
            e[13] = f[1] * d[0] + f[5] * d[1] + f[9] * d[2] + f[13];
            e[14] = f[2] * d[0] + f[6] * d[1] + f[10] * d[2] + f[14];
            e[15] = f[3] * d[0] + f[7] * d[1] + f[11] * d[2] + f[15];
            return e;
        };
        c.prototype.rotate = function (C, aa, B) {
            var ab = this.create();
            if (B.length != 3) {
                return undefined;
            }
            var V = Math.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
            if (!V) {
                return undefined;
            }
            var ag = B[0],
                af = B[1],
                ae = B[2];
            if (V != 1) {
                V = 1 / V;
                ag *= V;
                af *= V;
                ae *= V;
            }
            var ad = Math.sin(aa),
                ac = Math.cos(aa),
                Z = 1 - ac,
                Y = C[0],
                X = C[1],
                W = C[2],
                U = C[3],
                T = C[4],
                S = C[5],
                R = C[6],
                Q = C[7],
                P = C[8],
                O = C[9],
                M = C[10],
                L = C[11],
                K = ag * ag * Z + ac,
                J = af * ag * Z + ae * ad,
                I = ae * ag * Z - af * ad,
                H = ag * af * Z - ae * ad,
                G = af * af * Z + ac,
                F = ae * af * Z + ag * ad,
                E = ag * ae * Z + af * ad,
                D = af * ae * Z - ag * ad,
                N = ae * ae * Z + ac;
            if (aa) {
                if (C != ab) {
                    ab[12] = C[12];
                    ab[13] = C[13];
                    ab[14] = C[14];
                    ab[15] = C[15];
                }
            } else {
                ab = C;
            }
            ab[0] = Y * K + T * J + P * I;
            ab[1] = X * K + S * J + O * I;
            ab[2] = W * K + R * J + M * I;
            ab[3] = U * K + Q * J + L * I;
            ab[4] = Y * H + T * G + P * F;
            ab[5] = X * H + S * G + O * F;
            ab[6] = W * H + R * G + M * F;
            ab[7] = U * H + Q * G + L * F;
            ab[8] = Y * E + T * D + P * N;
            ab[9] = X * E + S * D + O * N;
            ab[10] = W * E + R * D + M * N;
            ab[11] = U * E + Q * D + L * N;
            return ab;
        };
        c.prototype.viewMatrix = function (m, x, j) {
            var t = this.identity(this.create());
            if (m.length != 3 || x.length != 3 || j.length != 3) {
                return undefined;
            }
            var r = m[0],
                q = m[1],
                p = m[2];
            var o = x[0],
                n = x[1],
                k = x[2];
            var w = j[0],
                v = j[1],
                u = j[2];
            if (r == o && q == n && p == k) {
                return t;
            }
            var i = r - o,
                h = q - n,
                g = p - k;
            var s = 1 / Math.sqrt(i * i + h * h + g * g);
            i *= s;
            h *= s;
            g *= s;
            var f = v * g - u * h;
            var e = u * i - w * g;
            var d = w * h - v * i;
            s = Math.sqrt(f * f + e * e + d * d);
            if (!s) {
                f = 0;
                e = 0;
                d = 0;
            } else {
                s = 1 / Math.sqrt(f * f + e * e + d * d);
                f *= s;
                e *= s;
                d *= s;
            }
            w = h * d - g * e;
            v = g * f - i * d;
            u = i * e - h * f;
            t[0] = f;
            t[1] = w;
            t[2] = i;
            t[3] = 0;
            t[4] = e;
            t[5] = v;
            t[6] = h;
            t[7] = 0;
            t[8] = d;
            t[9] = u;
            t[10] = g;
            t[11] = 0;
            t[12] = -(f * r + e * q + d * p);
            t[13] = -(w * r + v * q + u * p);
            t[14] = -(i * r + h * q + g * p);
            t[15] = 1;
            return t;
        };
        c.prototype.perspectiveMatrix = function (f, e, h, g) {
            var l = this.identity(this.create());
            var m = h * Math.tan((f * Math.PI) / 360);
            var d = m * e;
            var k = d * 2,
                j = m * 2,
                i = g - h;
            l[0] = (h * 2) / k;
            l[1] = 0;
            l[2] = 0;
            l[3] = 0;
            l[4] = 0;
            l[5] = (h * 2) / j;
            l[6] = 0;
            l[7] = 0;
            l[8] = 0;
            l[9] = 0;
            l[10] = -(g + h) / i;
            l[11] = -1;
            l[12] = 0;
            l[13] = 0;
            l[14] = -(g * h * 2) / i;
            l[15] = 0;
            return l;
        };
        c.prototype.orthoMatrix = function (f, n, l, e, j, i) {
            var o = this.identity(this.create());
            var g = n - f;
            var m = l - e;
            var k = i - j;
            o[0] = 2 / g;
            o[1] = 0;
            o[2] = 0;
            o[3] = 0;
            o[4] = 0;
            o[5] = 2 / m;
            o[6] = 0;
            o[7] = 0;
            o[8] = 0;
            o[9] = 0;
            o[10] = -2 / k;
            o[11] = 0;
            o[12] = -(f + n) / g;
            o[13] = -(l + e) / m;
            o[14] = -(i + j) / k;
            o[15] = 1;
            return o;
        };
        c.prototype.transpose = function (e) {
            var d = this.create();
            d[0] = e[0];
            d[1] = e[4];
            d[2] = e[8];
            d[3] = e[12];
            d[4] = e[1];
            d[5] = e[5];
            d[6] = e[9];
            d[7] = e[13];
            d[8] = e[2];
            d[9] = e[6];
            d[10] = e[10];
            d[11] = e[14];
            d[12] = e[3];
            d[13] = e[7];
            d[14] = e[11];
            d[15] = e[15];
            return d;
        };
        return c;
    })();
    a.WebGLMatrix = b;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var EcognitaMathLib;
(function (b) {
    var a = (function () {
        function c() {}
        c.prototype.create = function () {
            return new Float32Array(4);
        };
        c.prototype.identity = function (d) {
            d[0] = 0;
            d[1] = 0;
            d[2] = 0;
            d[3] = 1;
            return d;
        };
        c.prototype.inverse = function (e) {
            var d = this.create();
            d[0] = -e[0];
            d[1] = -e[1];
            d[2] = -e[2];
            d[3] = e[3];
            return d;
        };
        c.prototype.normalize = function (i) {
            var d = i[0],
                h = i[1],
                g = i[2],
                f = i[3];
            var e = Math.sqrt(d * d + h * h + g * g + f * f);
            if (e === 0) {
                i[0] = 0;
                i[1] = 0;
                i[2] = 0;
                i[3] = 0;
            } else {
                e = 1 / e;
                i[0] = d * e;
                i[1] = h * e;
                i[2] = g * e;
                i[3] = f * e;
            }
            return i;
        };
        c.prototype.multiply = function (h, g) {
            var e = this.create();
            var d = h[0],
                n = h[1],
                m = h[2],
                f = h[3];
            var k = g[0],
                j = g[1],
                i = g[2],
                l = g[3];
            e[0] = d * l + f * k + n * i - m * j;
            e[1] = n * l + f * j + m * k - d * i;
            e[2] = m * l + f * i + d * j - n * k;
            e[3] = f * l - d * k - n * j - m * i;
            return e;
        };
        c.prototype.rotate = function (j, h) {
            var i = Math.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
            if (!i) {
                console.log("need a axis value");
                return undefined;
            }
            var e = h[0],
                d = h[1],
                k = h[2];
            if (i != 1) {
                i = 1 / i;
                e *= i;
                d *= i;
                k *= i;
            }
            var g = Math.sin(j * 0.5);
            var f = this.create();
            f[0] = e * g;
            f[1] = d * g;
            f[2] = k * g;
            f[3] = Math.cos(j * 0.5);
            return f;
        };
        c.prototype.ToV3 = function (i, h) {
            var g = new Array(3);
            var e = this.inverse(h);
            var f = this.create();
            f[0] = i[0];
            f[1] = i[1];
            f[2] = i[2];
            var d = this.multiply(e, f);
            var j = this.multiply(d, h);
            g[0] = j[0];
            g[1] = j[1];
            g[2] = j[2];
            return g;
        };
        c.prototype.ToMat4x4 = function (n) {
            var p = new Float32Array(16);
            var j = n[0],
                i = n[1],
                h = n[2],
                k = n[3];
            var r = j + j,
                d = i + i,
                l = h + h;
            var g = j * r,
                f = j * d,
                e = j * l;
            var o = i * d,
                m = i * l,
                u = h * l;
            var v = k * r,
                t = k * d,
                s = k * l;
            p[0] = 1 - (o + u);
            p[1] = f - s;
            p[2] = e + t;
            p[3] = 0;
            p[4] = f + s;
            p[5] = 1 - (g + u);
            p[6] = m - v;
            p[7] = 0;
            p[8] = e - t;
            p[9] = m + v;
            p[10] = 1 - (g + o);
            p[11] = 0;
            p[12] = 0;
            p[13] = 0;
            p[14] = 0;
            p[15] = 1;
            return p;
        };
        c.prototype.slerp = function (g, f, e) {
            if (e < 0 || e > 1) {
                console.log("parameter time's setting is wrong!");
                return undefined;
            }
            var d = this.create();
            var k = g[0] * f[0] + g[1] * f[1] + g[2] * f[2] + g[3] * f[3];
            var l = 1 - k * k;
            if (l <= 0) {
                d[0] = g[0];
                d[1] = g[1];
                d[2] = g[2];
                d[3] = g[3];
            } else {
                l = Math.sqrt(l);
                if (Math.abs(l) < 0.0001) {
                    d[0] = g[0] * 0.5 + f[0] * 0.5;
                    d[1] = g[1] * 0.5 + f[1] * 0.5;
                    d[2] = g[2] * 0.5 + f[2] * 0.5;
                    d[3] = g[3] * 0.5 + f[3] * 0.5;
                } else {
                    var i = Math.acos(k);
                    var m = i * e;
                    var j = Math.sin(i - m) / l;
                    var h = Math.sin(m) / l;
                    d[0] = g[0] * j + f[0] * h;
                    d[1] = g[1] * j + f[1] * h;
                    d[2] = g[2] * j + f[2] * h;
                    d[3] = g[3] * j + f[3] * h;
                }
            }
            return d;
        };
        return c;
    })();
    b.WebGLQuaternion = a;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var EcognitaMathLib;
(function (e) {
    function i(j) {
        switch (j) {
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
    e.GetGLTypeSize = i;
    var d = (function () {
        function j(l, k, n, o, p, m) {
            if (o === void 0) {
                o = gl.REPEAT;
            }
            if (p === void 0) {
                p = gl.LINEAR;
            }
            if (m === void 0) {
                m = true;
            }
            this.type = k ? gl.FLOAT : gl.UNSIGNED_BYTE;
            this.format = [gl.LUMINANCE, gl.RG, gl.RGB, gl.RGBA][l - 1];
            this.glName = gl.createTexture();
            this.bind(this.glName);
            gl.texImage2D(gl.TEXTURE_2D, 0, this.format, this.format, this.type, n);
            if (m) {
                gl.generateMipmap(gl.TEXTURE_2D);
            }
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, p);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, p);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, o);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, o);
            this.texture = this.glName;
            this.bind(null);
        }
        j.prototype.bind = function (k) {
            gl.bindTexture(gl.TEXTURE_2D, k);
        };
        return j;
    })();
    e.WebGL_Texture = d;
    var h = (function () {
        function j(k) {
            this.cubeSource = k;
            this.cubeTarget = new Array(gl.TEXTURE_CUBE_MAP_POSITIVE_X, gl.TEXTURE_CUBE_MAP_POSITIVE_Y, gl.TEXTURE_CUBE_MAP_POSITIVE_Z, gl.TEXTURE_CUBE_MAP_NEGATIVE_X, gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, gl.TEXTURE_CUBE_MAP_NEGATIVE_Z);
            this.loadCubeTexture();
            this.cubeTexture = undefined;
        }
        j.prototype.loadCubeTexture = function () {
            var n = this;
            var k = new Array();
            var m = 0;
            this.cubeImage = k;
            for (var l = 0; l < this.cubeSource.length; l++) {
                k[l] = new Object();
                k[l].data = new Image();
                k[l].data.src = this.cubeSource[l];
                k[l].data.onload = function () {
                    m++;
                    if (m == n.cubeSource.length) {
                        n.generateCubeMap();
                    }
                };
            }
        };
        j.prototype.generateCubeMap = function () {
            var k = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, k);
            for (var l = 0; l < this.cubeSource.length; l++) {
                gl.texImage2D(this.cubeTarget[l], 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.cubeImage[l].data);
            }
            gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this.cubeTexture = k;
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
        };
        return j;
    })();
    e.WebGL_CubeMapTexture = h;
    var c = (function () {
        function j() {
            this.glName = gl.createFramebuffer();
        }
        j.prototype.bind = function () {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.glName);
        };
        j.prototype.unbind = function () {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        };
        j.prototype.attachTexture = function (l, k) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, gl.TEXTURE_2D, l.glName, 0);
        };
        j.prototype.detachTexture = function (k) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, gl.TEXTURE_2D, null, 0);
        };
        j.prototype.drawBuffers = function (m) {
            var k = [];
            for (var l = 0; l < m; ++l) {
                k.push(gl.COLOR_ATTACHMENT0 + l);
            }
            multiBufExt.drawBuffersWEBGL(k);
        };
        return j;
    })();
    e.WebGL_RenderTarget = c;
    var b = (function () {
        function j(l, k, m) {
            this.vertex = this.createShaderObject(l, k, false);
            this.fragment = this.createShaderObject(l, m, true);
            this.program = gl.createProgram();
            gl.attachShader(this.program, this.vertex);
            gl.attachShader(this.program, this.fragment);
            gl.linkProgram(this.program);
            this.uniforms = {};
            if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
                alert("Could not initialise shaders");
            }
        }
        j.prototype.bind = function () {
            gl.useProgram(this.program);
        };
        j.prototype.createShaderObject = function (m, l, q) {
            var p = this.resolveShaderSource(m, l);
            var o = gl.createShader(q ? gl.FRAGMENT_SHADER : gl.VERTEX_SHADER);
            gl.shaderSource(o, p);
            gl.compileShader(o);
            if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) {
                var k = p.split("\n");
                for (var n = 0; n < k.length; ++n) {
                    k[n] = ("   " + (n + 1)).slice(-4) + " | " + k[n];
                }
                p = k.join("\n");
                throw new Error((q ? "Fragment" : "Vertex") + " shader compilation error for shader '" + l + "':\n\n    " + gl.getShaderInfoLog(o).split("\n").join("\n    ") + "\nThe expanded shader source code was:\n\n" + p);
            }
            return o;
        };
        j.prototype.resolveShaderSource = function (m, l) {
            if (!(l in m)) {
                throw new Error("Unable to find shader source for '" + l + "'");
            }
            var o = m[l];
            var n = new RegExp('#include "(.+)"');
            var k;
            while ((k = n.exec(o))) {
                o = o.slice(0, k.index) + this.resolveShaderSource(m, k[1]) + o.slice(k.index + k[0].length);
            }
            return o;
        };
        j.prototype.uniformIndex = function (k) {
            if (!(k in this.uniforms)) {
                this.uniforms[k] = gl.getUniformLocation(this.program, k);
            }
            return this.uniforms[k];
        };
        j.prototype.uniformTexture = function (k, l) {
            var m = this.uniformIndex(k);
            if (m != -1) {
                gl.uniform1i(m, l.boundUnit);
            }
        };
        j.prototype.uniformF = function (k, l) {
            var m = this.uniformIndex(k);
            if (m != -1) {
                gl.uniform1f(m, l);
            }
        };
        j.prototype.uniform2F = function (l, k, n) {
            var m = this.uniformIndex(l);
            if (m != -1) {
                gl.uniform2f(m, k, n);
            }
        };
        return j;
    })();
    e.WebGL_Shader = b;
    var g = (function () {
        function j() {
            this.attributes = [];
            this.elementSize = 0;
        }
        j.prototype.bind = function (m) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.glName);
            for (var l = 0; l < this.attributes.length; ++l) {
                this.attributes[l].index = gl.getAttribLocation(m.program, this.attributes[l].name);
                if (this.attributes[l].index >= 0) {
                    var k = this.attributes[l];
                    gl.enableVertexAttribArray(k.index);
                    gl.vertexAttribPointer(k.index, k.size, k.type, k.norm, this.elementSize, k.offset);
                }
            }
        };
        j.prototype.release = function () {
            for (var k = 0; k < this.attributes.length; ++k) {
                if (this.attributes[k].index >= 0) {
                    gl.disableVertexAttribArray(this.attributes[k].index);
                    this.attributes[k].index = -1;
                }
            }
        };
        j.prototype.addAttribute = function (k, m, n, l) {
            this.attributes.push({ name: k, size: m, type: n, norm: l, offset: this.elementSize, index: -1 });
            this.elementSize += m * i(n);
        };
        j.prototype.addAttributes = function (k, l) {
            for (var m = 0; m < k.length; m++) {
                this.addAttribute(k[m], l[m], gl.FLOAT, false);
            }
        };
        j.prototype.init = function (k) {
            this.length = k;
            this.glName = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.glName);
            gl.bufferData(gl.ARRAY_BUFFER, this.length * this.elementSize, gl.STATIC_DRAW);
        };
        j.prototype.copy = function (k) {
            k = new Float32Array(k);
            if (k.byteLength != this.length * this.elementSize) {
                throw new Error("Resizing VBO during copy strongly discouraged");
            }
            gl.bufferData(gl.ARRAY_BUFFER, k, gl.STATIC_DRAW);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        };
        j.prototype.draw = function (l, k) {
            gl.drawArrays(l, 0, k ? k : this.length);
        };
        return j;
    })();
    e.WebGL_VertexBuffer = g;
    var a = (function () {
        function j() {}
        j.prototype.bind = function () {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.glName);
        };
        j.prototype.init = function (k) {
            this.length = k.length;
            this.glName = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.glName);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int16Array(k), gl.STATIC_DRAW);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        };
        j.prototype.draw = function (l, k) {
            gl.drawElements(l, k ? k : this.length, gl.UNSIGNED_SHORT, 0);
        };
        return j;
    })();
    e.WebGL_IndexBuffer = a;
    var f = (function () {
        function j(m, k) {
            this.width = m;
            this.height = k;
            var l = gl.createFramebuffer();
            this.framebuffer = l;
            var n = gl.createRenderbuffer();
            this.depthbuffer = n;
            var o = gl.createTexture();
            this.targetTexture = o;
        }
        j.prototype.bindFrameBuffer = function () {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        };
        j.prototype.bindDepthBuffer = function () {
            gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthbuffer);
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.width, this.height);
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthbuffer);
        };
        j.prototype.renderToTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        j.prototype.renderToShadowTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        j.prototype.renderToFloatTexure = function () {
            gl.bindTexture(gl.TEXTURE_2D, this.targetTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.FLOAT, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targetTexture, 0);
        };
        j.prototype.renderToCubeTexture = function (k) {
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.targetTexture);
            for (var l = 0; l < k.length; l++) {
                gl.texImage2D(k[l], 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        };
        j.prototype.releaseCubeTex = function () {
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
            gl.bindRenderbuffer(gl.RENDERBUFFER, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        };
        j.prototype.release = function () {
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.bindRenderbuffer(gl.RENDERBUFFER, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        };
        return j;
    })();
    e.WebGL_FrameBuffer = f;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var Shaders = {
    "Abstraction-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D akf;\nuniform sampler2D fxdog;\nuniform vec3 edge_color;\n\nuniform bool b_Abstraction;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n	vec2 uv = gl_FragCoord.xy / src_size ; \n    vec2 uv_src = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n    if(b_Abstraction){\n        vec2 d = 1.0 / src_size;\n        vec3 c = texture2D(akf, uv).xyz;\n        float e = texture2D(fxdog, uv).x;\n        gl_FragColor = vec4(mix(edge_color, c, e), 1.0);\n    }else{\n        gl_FragColor = texture2D(src, uv_src);\n    }\n}\n",
    "Abstraction-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "AKF-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D k0;\nuniform sampler2D tfm;\nuniform float radius;\nuniform float q;\nuniform float alpha;\n\nuniform bool anisotropic;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nconst float PI = 3.14159265358979323846;\nconst int N = 8;\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n	vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n\n    if(anisotropic){\n        vec4 m[8];\n        vec3 s[8];\n        for (int k = 0; k < N; ++k) {\n            m[k] = vec4(0.0);\n            s[k] = vec3(0.0);\n        }\n\n        float piN = 2.0 * PI / float(N);\n        mat2 X = mat2(cos(piN), sin(piN), -sin(piN), cos(piN));\n\n        vec4 t = texture2D(tfm, uv);\n        float a = radius * clamp((alpha + t.w) / alpha, 0.1, 2.0); \n        float b = radius * clamp(alpha / (alpha + t.w), 0.1, 2.0);\n\n        float cos_phi = cos(t.z);\n        float sin_phi = sin(t.z);\n\n        mat2 R = mat2(cos_phi, -sin_phi, sin_phi, cos_phi);\n        mat2 S = mat2(0.5/a, 0.0, 0.0, 0.5/b);\n        mat2 SR = S * R;\n\n        const int max_x = 6;\n        const int max_y = 6;\n\n        for (int j = -max_y; j <= max_y; ++j) {\n            for (int i = -max_x; i <= max_x; ++i) {\n                vec2 v = SR * vec2(i,j);\n                if (dot(v,v) <= 0.25) {\n                vec4 c_fix = texture2D(src, uv + vec2(i,j) / src_size);\n                vec3 c = c_fix.rgb;\n                for (int k = 0; k < N; ++k) {\n                    float w = texture2D(k0, vec2(0.5, 0.5) + v).x;\n\n                    m[k] += vec4(c * w, w);\n                    s[k] += c * c * w;\n\n                    v *= X;\n                    }\n                }\n            }\n        }\n\n        vec4 o = vec4(0.0);\n        for (int k = 0; k < N; ++k) {\n            m[k].rgb /= m[k].w;\n            s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);\n\n            float sigma2 = s[k].r + s[k].g + s[k].b;\n            float w = 1.0 / (1.0 + pow(255.0 * sigma2, 0.5 * q));\n\n            o += vec4(m[k].rgb * w, w);\n        }\n\n        gl_FragColor = vec4(o.rgb / o.w, 1.0);\n    }else{\n        gl_FragColor = texture2D(src, uv);\n    }\n\n}\n",
    "AKF-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "Anisotropic-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\nuniform sampler2D visual;\nuniform bool anisotropic;\nuniform float cvsHeight;\nuniform float cvsWidth;\nvarying vec2 vTexCoord;\n\nvoid main (void) {\n	vec2 src_size = vec2(cvsWidth, cvsHeight);\n	vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n	vec4 t = texture2D( tfm, uv );\n\n	if(anisotropic){\n		gl_FragColor = texture2D(visual, vec2(t.w,0.5));\n	}else{\n		gl_FragColor = texture2D(src, uv);\n	}\n}\n",
    "Anisotropic-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "blurEffect-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nvarying vec4      vColor;\n\nvoid main(void){\n	vec2 tFrag = vec2(1.0 / 512.0);\n	vec4 destColor = texture2D(texture, gl_FragCoord.st * tFrag);\n	destColor *= 0.36;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0,  1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  0.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  0.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0, -1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0, -1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0, -1.0)) * tFrag) * 0.04;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0,  2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0,  2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0,  2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  1.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  1.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0,  0.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0,  0.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0, -1.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0, -1.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-2.0, -2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2(-1.0, -2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 0.0, -2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 1.0, -2.0)) * tFrag) * 0.02;\n	destColor += texture2D(texture, (gl_FragCoord.st + vec2( 2.0, -2.0)) * tFrag) * 0.02;\n\n	gl_FragColor = vColor * destColor;\n}\n",
    "blurEffect-vert": "attribute vec3 position;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vColor      = color;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "bumpMapping-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nvarying vec4      vColor;\nvarying vec2      vTextureCoord;\nvarying vec3      vEyeDirection;\nvarying vec3      vLightDirection;\n\nvoid main(void){\n	vec3 mNormal    = (texture2D(texture, vTextureCoord) * 2.0 - 1.0).rgb;\n	vec3 light      = normalize(vLightDirection);\n	vec3 eye        = normalize(vEyeDirection);\n	vec3 halfLE     = normalize(light + eye);\n	float diffuse   = clamp(dot(mNormal, light), 0.1, 1.0);\n	float specular  = pow(clamp(dot(mNormal, halfLE), 0.0, 1.0), 50.0);\n	vec4  destColor = vColor * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0);\n	gl_FragColor    = destColor;\n}\n",
    "bumpMapping-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nattribute vec2 textureCoord;\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nuniform   mat4 invMatrix;\nuniform   vec3 lightPosition;\nuniform   vec3 eyePosition;\nvarying   vec4 vColor;\nvarying   vec2 vTextureCoord;\nvarying   vec3 vEyeDirection;\nvarying   vec3 vLightDirection;\n\nvoid main(void){\n	vec3 pos      = (mMatrix * vec4(position, 0.0)).xyz;\n	vec3 invEye   = (invMatrix * vec4(eyePosition, 0.0)).xyz;\n	vec3 invLight = (invMatrix * vec4(lightPosition, 0.0)).xyz;\n	vec3 eye      = invEye - pos;\n	vec3 light    = invLight - pos;\n	vec3 n = normalize(normal);\n	vec3 t = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));\n	vec3 b = cross(n, t);\n	vEyeDirection.x   = dot(t, eye);\n	vEyeDirection.y   = dot(b, eye);\n	vEyeDirection.z   = dot(n, eye);\n	normalize(vEyeDirection);\n	vLightDirection.x = dot(t, light);\n	vLightDirection.y = dot(b, light);\n	vLightDirection.z = dot(n, light);\n	normalize(vLightDirection);\n	vColor         = color;\n	vTextureCoord  = textureCoord;\n	gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "cubeTexBumpMapping-frag":
        "precision mediump float;\n\nuniform vec3        eyePosition;\nuniform sampler2D   normalMap;\nuniform samplerCube cubeTexture;\nuniform bool        reflection;\nvarying vec3        vPosition;\nvarying vec2        vTextureCoord;\nvarying vec3        vNormal;\nvarying vec3        tTangent;\n\nvarying vec4        vColor;\n\n//reflect = I - 2.0 * dot(N, I) * N.\nvec3 egt_reflect(vec3 p, vec3 n){\n  return  p - 2.0* dot(n,p) * n;\n}\n\nvoid main(void){\n	vec3 tBinormal = cross(vNormal, tTangent);\n	mat3 mView     = mat3(tTangent, tBinormal, vNormal);\n	vec3 mNormal   = mView * (texture2D(normalMap, vTextureCoord) * 2.0 - 1.0).rgb;\n	vec3 ref;\n	if(reflection){\n		ref = reflect(vPosition - eyePosition, mNormal);\n        //ref = egt_reflect(normalize(vPosition - eyePosition),normalize(vNormal));\n	}else{\n		ref = vNormal;\n	}\n	vec4 envColor  = textureCube(cubeTexture, ref);\n	vec4 destColor = vColor * envColor;\n	gl_FragColor   = destColor;\n}\n",
    "cubeTexBumpMapping-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nattribute vec2 textureCoord;\n\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nvarying   vec3 vPosition;\nvarying   vec2 vTextureCoord;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\nvarying   vec3 tTangent;\n\nvoid main(void){\n	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n	vNormal     = (mMatrix * vec4(normal, 0.0)).xyz;\n	vTextureCoord = textureCoord;\n	vColor      = color;\n	tTangent      = cross(vNormal, vec3(0.0, 1.0, 0.0));\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "cubeTexMapping-frag":
        "precision mediump float;\n\nuniform vec3        eyePosition;\nuniform samplerCube cubeTexture;\nuniform bool        reflection;\nvarying vec3        vPosition;\nvarying vec3        vNormal;\nvarying vec4        vColor;\n\n//reflect = I - 2.0 * dot(N, I) * N.\nvec3 egt_reflect(vec3 p, vec3 n){\n  return  p - 2.0* dot(n,p) * n;\n}\n\nvoid main(void){\n	vec3 ref;\n	if(reflection){\n		ref = reflect(vPosition - eyePosition, vNormal);\n        //ref = egt_reflect(normalize(vPosition - eyePosition),normalize(vNormal));\n	}else{\n		ref = vNormal;\n	}\n	vec4 envColor  = textureCube(cubeTexture, ref);\n	vec4 destColor = vColor * envColor;\n	gl_FragColor   = destColor;\n}\n",
    "cubeTexMapping-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nvarying   vec3 vPosition;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n	vNormal     = (mMatrix * vec4(normal, 0.0)).xyz;\n	vColor      = color;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "demo-frag": "void main(void){\n	gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n}\n",
    "demo-vert": "attribute vec3 position;\nuniform   mat4 mvpMatrix;\n\nvoid main(void){\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "demo1-frag": "precision mediump float;\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "demo1-vert": "attribute vec3 position;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nvarying vec4 vColor;\n\nvoid main(void){\n	vColor = color;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "directionLighting-frag": "precision mediump float;\n\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "directionLighting-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec3 normal;\n\nuniform mat4 mvpMatrix;\nuniform mat4 invMatrix;\nuniform vec3 lightDirection;\nvarying vec4 vColor;\n\nvoid main(void){\n    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0)).xyz;\n    float diffuse = clamp(dot(invLight,normal),0.1,1.0);\n    vColor = color*vec4(vec3(diffuse),1.0);\n    gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "dir_ambient-frag": "precision mediump float;\n\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "dir_ambient-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec3 normal;\n\nuniform mat4 mvpMatrix;\nuniform mat4 invMatrix;\nuniform vec3 lightDirection;\nuniform vec4 ambientColor;\nvarying vec4 vColor;\n\nvoid main(void){\n    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0)).xyz;\n    float diffuse = clamp(dot(invLight,normal),0.1,1.0);\n    vColor = color*vec4(vec3(diffuse),1.0) +ambientColor;\n    gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "DoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\n\nuniform bool b_DoG;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma_e;\nuniform float sigma_r;\nuniform float tau;\nuniform float phi;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n    vec3 destColor = vec3(0.0);\n    if(b_DoG){\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2 uv = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n        float twoSigmaESquared = 2.0 * sigma_e * sigma_e;\n        float twoSigmaRSquared = 2.0 * sigma_r * sigma_r;\n        int halfWidth = int(ceil( 2.0 * sigma_r ));\n\n        const int MAX_NUM_ITERATION = 99999;\n        vec2 sum = vec2(0.0);\n        vec2 norm = vec2(0.0);\n\n        for(int cnt=0;cnt<MAX_NUM_ITERATION;cnt++){\n            if(cnt > (2*halfWidth+1)*(2*halfWidth+1)){break;}\n            int i = int(cnt / (2*halfWidth+1)) - halfWidth;\n            int j = cnt - halfWidth - int(cnt / (2*halfWidth+1)) * (2*halfWidth+1);\n\n            float d = length(vec2(i,j));\n            vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n                                exp( -d * d / twoSigmaRSquared ));\n\n            vec2 L = texture2D(src, (uv + vec2(i,j)) * Frag).xx;\n\n            norm += 2.0 * kernel;\n            sum += kernel * L;\n        }\n\n        sum /= norm;\n\n        float H = 100.0 * (sum.x - tau * sum.y);\n        float edge = ( H > 0.0 )? 1.0 : 2.0 * smoothstep(-2.0, 2.0, phi * H );\n        destColor = vec3(edge);\n    }else{\n        destColor = texture2D(src, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "DoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "ETF-frag":
        "// Edge Tangent Flow\nprecision mediump float;\n\nuniform sampler2D src;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 uv = gl_FragCoord.xy / src_size;\n    vec2 d = 1.0 / src_size;\n    vec3 c = texture2D(src, uv).xyz;\n    float gx = c.z;\n    vec2 tx = c.xy;\n    const float KERNEL = 5.0;\n    vec2 etf = vec2(0.0);\n    vec2 sum = vec2(0,0);\n    float weight = 0.0;\n\n    for(float j = -KERNEL ; j<KERNEL;j++){\n        for(float i=-KERNEL ; i<KERNEL;i++){\n            vec2 ty = texture2D(src, uv + vec2(i * d.x, j * d.y)).xy;\n            float gy = texture2D(src, uv + vec2(i * d.x, j * d.y)).z;\n\n            float wd = abs(dot(tx,ty));\n            float wm = (gy - gx + 1.0)/2.0;\n            float phi = dot(gx,gy)>0.0?1.0:-1.0;\n            float ws = sqrt(j*j+i*i) < KERNEL?1.0:0.0;\n\n            sum += ty * (wm * wd );\n            weight += wm * wd ;\n        }\n    }\n\n    if(weight != 0.0){\n        etf = sum / weight;\n    }else{\n        etf = vec2(0.0);\n    }\n\n    float mag = sqrt(etf.x*etf.x + etf.y*etf.y);\n    float vx = etf.x/mag;\n    float vy = etf.y/mag;\n    gl_FragColor = vec4(vx,vy,mag, 1.0);\n}\n",
    "ETF-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "FDoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\n\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma_m;\nuniform float phi;\n\nuniform bool b_FDoG;\nvarying vec2 vTexCoord;\n\nstruct lic_t { \n    vec2 p; \n    vec2 t;\n    float w;\n    float dw;\n};\n\nvoid step(inout lic_t s) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 t = texture2D(tfm, s.p).xy;\n    if (dot(t, s.t) < 0.0) t = -t;\n    s.t = t;\n\n    s.dw = (abs(t.x) > abs(t.y))? \n        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n    s.p += t * s.dw / src_size;\n    s.w += s.dw;\n}\n\nvoid main (void) {\n\n    vec3 destColor = vec3(0.0);\n    if(b_FDoG){\n        vec2 src_size = vec2(cvsWidth, cvsHeight);\n        vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n\n        float twoSigmaMSquared = 2.0 * sigma_m * sigma_m;\n        float halfWidth = 2.0 * sigma_m;\n\n        float H = texture2D( src, uv ).x;\n        float w = 1.0;\n\n        lic_t a, b;\n        a.p = b.p = uv;\n        a.t = texture2D( tfm, uv ).xy / src_size;\n        b.t = -a.t;\n        a.w = b.w = 0.0;\n\n        const int MAX_NUM_ITERATION = 99999;\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (a.w < halfWidth) {\n                step(a);\n                float k = a.dw * exp(-a.w * a.w / twoSigmaMSquared);\n                H += k * texture2D(src, a.p).x;\n                w += k;\n            }else{\n                break;\n            }\n        }\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (b.w < halfWidth) {\n                step(b);\n                float k = b.dw * exp(-b.w * b.w / twoSigmaMSquared);\n                H += k * texture2D(src, b.p).x;\n                w += k;\n            }else{\n                break;\n            }\n        }\n        H /= w;\n        float edge = ( H > 0.0 )? 1.0 : 2.0 * smoothstep(-2.0, 2.0, phi * H );\n        destColor = vec3(edge);\n    }\n    else{\n        destColor = texture2D(src, vTexCoord).rgb;\n    }\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "FDoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "filterScene-frag": "precision mediump float;\n\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "filterScene-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nuniform   mat4 invMatrix;\nuniform   vec3 lightDirection;\nuniform   vec3 eyeDirection;\nuniform   vec4 ambientColor;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n	vec3  invEye   = normalize(invMatrix * vec4(eyeDirection, 0.0)).xyz;\n	vec3  halfLE   = normalize(invLight + invEye);\n	float diffuse  = clamp(dot(normal, invLight), 0.0, 1.0);\n	float specular = pow(clamp(dot(normal, halfLE), 0.0, 1.0), 50.0);\n	vec4  amb      = color * ambientColor;\n	vColor         = amb * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0);\n	gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "frameBuffer-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nvarying vec4      vColor;\nvarying vec2      vTextureCoord;\n\nvoid main(void){\n	vec4 smpColor = texture2D(texture, vTextureCoord);\n	gl_FragColor  = vColor * smpColor;\n}\n",
    "frameBuffer-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nattribute vec2 textureCoord;\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nuniform   mat4 invMatrix;\nuniform   vec3 lightDirection;\nuniform   bool useLight;\nvarying   vec4 vColor;\nvarying   vec2 vTextureCoord;\n\nvoid main(void){\n	if(useLight){\n		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n		float diffuse  = clamp(dot(normal, invLight), 0.2, 1.0);\n		vColor         = vec4(color.xyz * vec3(diffuse), 1.0);\n	}else{\n		vColor         = color;\n	}\n	vTextureCoord  = textureCoord;\n	gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "FXDoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\n\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma_m;\nuniform float phi;\nuniform float epsilon;\n\nuniform bool b_FXDoG;\nvarying vec2 vTexCoord;\n\nfloat cosh(float val)\n{\n    float tmp = exp(val);\n    float cosH = (tmp + 1.0 / tmp) / 2.0;\n    return cosH;\n}\n\nfloat tanh(float val)\n{\n    float tmp = exp(val);\n    float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n    return tanH;\n}\n\nfloat sinh(float val)\n{\n    float tmp = exp(val);\n    float sinH = (tmp - 1.0 / tmp) / 2.0;\n    return sinH;\n}\n\nstruct lic_t { \n    vec2 p; \n    vec2 t;\n    float w;\n    float dw;\n};\n\nvoid step(inout lic_t s) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 t = texture2D(tfm, s.p).xy;\n    if (dot(t, s.t) < 0.0) t = -t;\n    s.t = t;\n\n    s.dw = (abs(t.x) > abs(t.y))? \n        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n    s.p += t * s.dw / src_size;\n    s.w += s.dw;\n}\n\nvoid main (void) {\n\n    vec3 destColor = vec3(0.0);\n    if(b_FXDoG){\n        vec2 src_size = vec2(cvsWidth, cvsHeight);\n        vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n\n        float twoSigmaMSquared = 2.0 * sigma_m * sigma_m;\n        float halfWidth = 2.0 * sigma_m;\n\n        float H = texture2D( src, uv ).x;\n        float w = 1.0;\n\n        lic_t a, b;\n        a.p = b.p = uv;\n        a.t = texture2D( tfm, uv ).xy / src_size;\n        b.t = -a.t;\n        a.w = b.w = 0.0;\n\n        const int MAX_NUM_ITERATION = 99999;\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (a.w < halfWidth) {\n                step(a);\n                float k = a.dw * exp(-a.w * a.w / twoSigmaMSquared);\n                H += k * texture2D(src, a.p).x;\n                w += k;\n            }else{\n                break;\n            }\n        }\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (b.w < halfWidth) {\n                step(b);\n                float k = b.dw * exp(-b.w * b.w / twoSigmaMSquared);\n                H += k * texture2D(src, b.p).x;\n                w += k;\n            }else{\n                break;\n            }\n        }\n        H /= w;\n        float edge = ( H > epsilon )? 1.0 : 1.0 + tanh( phi * (H - epsilon));\n        destColor = vec3(edge);\n    }\n    else{\n        destColor = texture2D(src, vTexCoord).rgb;\n    }\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "FXDoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "gaussianFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nuniform bool b_gaussian;\nuniform float cvsHeight;\nuniform float cvsWidth;\nuniform float weight[10];\nuniform bool horizontal;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n    vec3  destColor = vec3(0.0);\n	if(b_gaussian){\n		float tFrag = 1.0 / cvsHeight;\n		float sFrag = 1.0 / cvsWidth;\n		vec2  Frag = vec2(sFrag,tFrag);\n		vec2 fc;\n		if(horizontal){\n			fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n			destColor += texture2D(texture, (fc + vec2(-9.0, 0.0)) * Frag).rgb * weight[9];\n			destColor += texture2D(texture, (fc + vec2(-8.0, 0.0)) * Frag).rgb * weight[8];\n			destColor += texture2D(texture, (fc + vec2(-7.0, 0.0)) * Frag).rgb * weight[7];\n			destColor += texture2D(texture, (fc + vec2(-6.0, 0.0)) * Frag).rgb * weight[6];\n			destColor += texture2D(texture, (fc + vec2(-5.0, 0.0)) * Frag).rgb * weight[5];\n			destColor += texture2D(texture, (fc + vec2(-4.0, 0.0)) * Frag).rgb * weight[4];\n			destColor += texture2D(texture, (fc + vec2(-3.0, 0.0)) * Frag).rgb * weight[3];\n			destColor += texture2D(texture, (fc + vec2(-2.0, 0.0)) * Frag).rgb * weight[2];\n			destColor += texture2D(texture, (fc + vec2(-1.0, 0.0)) * Frag).rgb * weight[1];\n			destColor += texture2D(texture, (fc + vec2( 0.0, 0.0)) * Frag).rgb * weight[0];\n			destColor += texture2D(texture, (fc + vec2( 1.0, 0.0)) * Frag).rgb * weight[1];\n			destColor += texture2D(texture, (fc + vec2( 2.0, 0.0)) * Frag).rgb * weight[2];\n			destColor += texture2D(texture, (fc + vec2( 3.0, 0.0)) * Frag).rgb * weight[3];\n			destColor += texture2D(texture, (fc + vec2( 4.0, 0.0)) * Frag).rgb * weight[4];\n			destColor += texture2D(texture, (fc + vec2( 5.0, 0.0)) * Frag).rgb * weight[5];\n			destColor += texture2D(texture, (fc + vec2( 6.0, 0.0)) * Frag).rgb * weight[6];\n			destColor += texture2D(texture, (fc + vec2( 7.0, 0.0)) * Frag).rgb * weight[7];\n			destColor += texture2D(texture, (fc + vec2( 8.0, 0.0)) * Frag).rgb * weight[8];\n			destColor += texture2D(texture, (fc + vec2( 9.0, 0.0)) * Frag).rgb * weight[9];\n		}else{\n			fc = gl_FragCoord.st;\n			destColor += texture2D(texture, (fc + vec2(0.0, -9.0)) * Frag).rgb * weight[9];\n			destColor += texture2D(texture, (fc + vec2(0.0, -8.0)) * Frag).rgb * weight[8];\n			destColor += texture2D(texture, (fc + vec2(0.0, -7.0)) * Frag).rgb * weight[7];\n			destColor += texture2D(texture, (fc + vec2(0.0, -6.0)) * Frag).rgb * weight[6];\n			destColor += texture2D(texture, (fc + vec2(0.0, -5.0)) * Frag).rgb * weight[5];\n			destColor += texture2D(texture, (fc + vec2(0.0, -4.0)) * Frag).rgb * weight[4];\n			destColor += texture2D(texture, (fc + vec2(0.0, -3.0)) * Frag).rgb * weight[3];\n			destColor += texture2D(texture, (fc + vec2(0.0, -2.0)) * Frag).rgb * weight[2];\n			destColor += texture2D(texture, (fc + vec2(0.0, -1.0)) * Frag).rgb * weight[1];\n			destColor += texture2D(texture, (fc + vec2(0.0,  0.0)) * Frag).rgb * weight[0];\n			destColor += texture2D(texture, (fc + vec2(0.0,  1.0)) * Frag).rgb * weight[1];\n			destColor += texture2D(texture, (fc + vec2(0.0,  2.0)) * Frag).rgb * weight[2];\n			destColor += texture2D(texture, (fc + vec2(0.0,  3.0)) * Frag).rgb * weight[3];\n			destColor += texture2D(texture, (fc + vec2(0.0,  4.0)) * Frag).rgb * weight[4];\n			destColor += texture2D(texture, (fc + vec2(0.0,  5.0)) * Frag).rgb * weight[5];\n			destColor += texture2D(texture, (fc + vec2(0.0,  6.0)) * Frag).rgb * weight[6];\n			destColor += texture2D(texture, (fc + vec2(0.0,  7.0)) * Frag).rgb * weight[7];\n			destColor += texture2D(texture, (fc + vec2(0.0,  8.0)) * Frag).rgb * weight[8];\n			destColor += texture2D(texture, (fc + vec2(0.0,  9.0)) * Frag).rgb * weight[9];\n		}\n	}else{\n 		destColor = texture2D(texture, vTexCoord).rgb;\n	}\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "gaussianFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "Gaussian_K-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform float sigma;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 uv = gl_FragCoord.xy / src_size;\n\n    float twoSigma2 = 2.0 * 2.0 * 2.0;\n    const int halfWidth = 4;//int(ceil( 2.0 * sigma ));\n\n    vec3 sum = vec3(0.0);\n    float norm = 0.0;\n    for ( int i = -halfWidth; i <= halfWidth; ++i ) {\n        for ( int j = -halfWidth; j <= halfWidth; ++j ) {\n            float d = length(vec2(i,j));\n            float kernel = exp( -d *d / twoSigma2 );\n            vec3 c = texture2D(src, uv + vec2(i,j) / src_size ).rgb;\n            sum += kernel * c;\n            norm += kernel;\n        }\n    }\n    gl_FragColor = vec4(sum / norm, 1.0);\n}\n",
    "Gaussian_K-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "gkuwaharaFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\n\nuniform float weight[49];\nuniform bool b_gkuwahara;\nuniform float cvsHeight;\nuniform float cvsWidth;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n    vec3  destColor = vec3(0.0);\n    if(b_gkuwahara){\n        float q = 3.0;\n        vec3 mean[8];\n        vec3 sigma[8];\n        vec2 offset[49];\n        offset[0] = vec2(-3.0, -3.0);\n        offset[1] = vec2(-2.0, -3.0);\n        offset[2] = vec2(-1.0, -3.0);\n        offset[3] = vec2( 0.0, -3.0);\n        offset[4] = vec2( 1.0, -3.0);\n        offset[5] = vec2( 2.0, -3.0);\n        offset[6] = vec2( 3.0, -3.0);\n\n        offset[7]  = vec2(-3.0, -2.0);\n        offset[8]  = vec2(-2.0, -2.0);\n        offset[9]  = vec2(-1.0, -2.0);\n        offset[10] = vec2( 0.0, -2.0);\n        offset[11] = vec2( 1.0, -2.0);\n        offset[12] = vec2( 2.0, -2.0);\n        offset[13] = vec2( 3.0, -2.0);\n\n        offset[14] = vec2(-3.0, -1.0);\n        offset[15] = vec2(-2.0, -1.0);\n        offset[16] = vec2(-1.0, -1.0);\n        offset[17] = vec2( 0.0, -1.0);\n        offset[18] = vec2( 1.0, -1.0);\n        offset[19] = vec2( 2.0, -1.0);\n        offset[20] = vec2( 3.0, -1.0);\n\n        offset[21] = vec2(-3.0,  0.0);\n        offset[22] = vec2(-2.0,  0.0);\n        offset[23] = vec2(-1.0,  0.0);\n        offset[24] = vec2( 0.0,  0.0);\n        offset[25] = vec2( 1.0,  0.0);\n        offset[26] = vec2( 2.0,  0.0);\n        offset[27] = vec2( 3.0,  0.0);\n\n        offset[28] = vec2(-3.0,  1.0);\n        offset[29] = vec2(-2.0,  1.0);\n        offset[30] = vec2(-1.0,  1.0);\n        offset[31] = vec2( 0.0,  1.0);\n        offset[32] = vec2( 1.0,  1.0);\n        offset[33] = vec2( 2.0,  1.0);\n        offset[34] = vec2( 3.0,  1.0);\n\n        offset[35] = vec2(-3.0,  2.0);\n        offset[36] = vec2(-2.0,  2.0);\n        offset[37] = vec2(-1.0,  2.0);\n        offset[38] = vec2( 0.0,  2.0);\n        offset[39] = vec2( 1.0,  2.0);\n        offset[40] = vec2( 2.0,  2.0);\n        offset[41] = vec2( 3.0,  2.0);\n\n        offset[42] = vec2(-3.0,  3.0);\n        offset[43] = vec2(-2.0,  3.0);\n        offset[44] = vec2(-1.0,  3.0);\n        offset[45] = vec2( 0.0,  3.0);\n        offset[46] = vec2( 1.0,  3.0);\n        offset[47] = vec2( 2.0,  3.0);\n        offset[48] = vec2( 3.0,  3.0);\n\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n        vec3 cur_std = vec3(0.0);\n        float cur_weight = 0.0;\n        vec3 total_ms = vec3(0.0);\n        vec3 total_s = vec3(0.0);\n\n        mean[0]=vec3(0.0);\n        sigma[0]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * weight[24];\n        sigma[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2D(texture, (fc + offset[24]) * Frag).rgb * weight[24];\n        cur_weight+= weight[24];\n        mean[0]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * weight[31];\n        sigma[0]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2D(texture, (fc + offset[31]) * Frag).rgb * weight[31];\n        cur_weight+= weight[31];\n        mean[0]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * weight[38];\n        sigma[0]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2D(texture, (fc + offset[38]) * Frag).rgb * weight[38];\n        cur_weight+= weight[38];\n        mean[0]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * weight[45];\n        sigma[0]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2D(texture, (fc + offset[45]) * Frag).rgb * weight[45];\n        cur_weight+= weight[45];\n        mean[0]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * weight[39];\n        sigma[0]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * texture2D(texture, (fc + offset[39]) * Frag).rgb * weight[39];\n        cur_weight+= weight[39];\n        mean[0]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * weight[46];\n        sigma[0]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * texture2D(texture, (fc + offset[46]) * Frag).rgb * weight[46];\n        cur_weight+= weight[46];\n        mean[0]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * weight[47];\n        sigma[0]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * texture2D(texture, (fc + offset[47]) * Frag).rgb * weight[47];\n        cur_weight+= weight[47];\n\n        if(cur_weight!=0.0){\n            mean[0] /= cur_weight;\n            sigma[0] /= cur_weight;\n        }\n\n        cur_std = sigma[0] - mean[0] * mean[0];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[0] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[1]=vec3(0.0);\n        sigma[1]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[1]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * weight[32];\n        sigma[1]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * texture2D(texture, (fc + offset[32]) * Frag).rgb * weight[32];\n        cur_weight+= weight[32];\n        mean[1]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * weight[33];\n        sigma[1]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * texture2D(texture, (fc + offset[33]) * Frag).rgb * weight[33];\n        cur_weight+= weight[33];\n        mean[1]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * weight[40];\n        sigma[1]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * texture2D(texture, (fc + offset[40]) * Frag).rgb * weight[40];\n        cur_weight+= weight[40];\n        mean[1]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * weight[34];\n        sigma[1]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * texture2D(texture, (fc + offset[34]) * Frag).rgb * weight[34];\n        cur_weight+= weight[34];\n        mean[1]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * weight[41];\n        sigma[1]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * texture2D(texture, (fc + offset[41]) * Frag).rgb * weight[41];\n        cur_weight+= weight[41];\n        mean[1]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * weight[48];\n        sigma[1]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * texture2D(texture, (fc + offset[48]) * Frag).rgb * weight[48];\n        cur_weight+= weight[48];\n\n        if(cur_weight!=0.0){\n            mean[1] /= cur_weight;\n            sigma[1] /= cur_weight;\n        }\n\n        cur_std = sigma[1] - mean[1] * mean[1];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[1] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[2]=vec3(0.0);\n        sigma[2]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[2]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * weight[25];\n        sigma[2]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2D(texture, (fc + offset[25]) * Frag).rgb * weight[25];\n        cur_weight+= weight[25];\n        mean[2]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * weight[19];\n        sigma[2]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * texture2D(texture, (fc + offset[19]) * Frag).rgb * weight[19];\n        cur_weight+= weight[19];\n        mean[2]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * weight[26];\n        sigma[2]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2D(texture, (fc + offset[26]) * Frag).rgb * weight[26];\n        cur_weight+= weight[26];\n        mean[2]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * weight[13];\n        sigma[2]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * texture2D(texture, (fc + offset[13]) * Frag).rgb * weight[13];\n        cur_weight+= weight[13];\n        mean[2]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * weight[20];\n        sigma[2]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * texture2D(texture, (fc + offset[20]) * Frag).rgb * weight[20];\n        cur_weight+= weight[20];\n        mean[2]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * weight[27];\n        sigma[2]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2D(texture, (fc + offset[27]) * Frag).rgb * weight[27];\n        cur_weight+= weight[27];\n\n        if(cur_weight!=0.0){\n            mean[2] /= cur_weight;\n            sigma[2] /= cur_weight;\n        }\n\n        cur_std = sigma[2] - mean[2] * mean[2];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[2] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[3]=vec3(0.0);\n        sigma[3]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[3]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * weight[4];\n        sigma[3]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * texture2D(texture, (fc + offset[4]) * Frag).rgb * weight[4];\n        cur_weight+= weight[4];\n        mean[3]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * weight[11];\n        sigma[3]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * texture2D(texture, (fc + offset[11]) * Frag).rgb * weight[11];\n        cur_weight+= weight[11];\n        mean[3]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * weight[18];\n        sigma[3]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * texture2D(texture, (fc + offset[18]) * Frag).rgb * weight[18];\n        cur_weight+= weight[18];\n        mean[3]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * weight[5];\n        sigma[3]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * texture2D(texture, (fc + offset[5]) * Frag).rgb * weight[5];\n        cur_weight+= weight[5];\n        mean[3]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * weight[12];\n        sigma[3]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * texture2D(texture, (fc + offset[12]) * Frag).rgb * weight[12];\n        cur_weight+= weight[12];\n        mean[3]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * weight[6];\n        sigma[3]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * texture2D(texture, (fc + offset[6]) * Frag).rgb * weight[6];\n        cur_weight+= weight[6];\n\n        if(cur_weight!=0.0){\n            mean[3] /= cur_weight;\n            sigma[3] /= cur_weight;\n        }\n\n        cur_std = sigma[3] - mean[3] * mean[3];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[3] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[4]=vec3(0.0);\n        sigma[4]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[4]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * weight[1];\n        sigma[4]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * texture2D(texture, (fc + offset[1]) * Frag).rgb * weight[1];\n        cur_weight+= weight[1];\n        mean[4]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * weight[2];\n        sigma[4]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * texture2D(texture, (fc + offset[2]) * Frag).rgb * weight[2];\n        cur_weight+= weight[2];\n        mean[4]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * weight[9];\n        sigma[4]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * texture2D(texture, (fc + offset[9]) * Frag).rgb * weight[9];\n        cur_weight+= weight[9];\n        mean[4]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * weight[3];\n        sigma[4]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D(texture, (fc + offset[3]) * Frag).rgb * weight[3];\n        cur_weight+= weight[3];\n        mean[4]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * weight[10];\n        sigma[4]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2D(texture, (fc + offset[10]) * Frag).rgb * weight[10];\n        cur_weight+= weight[10];\n        mean[4]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * weight[17];\n        sigma[4]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2D(texture, (fc + offset[17]) * Frag).rgb * weight[17];\n        cur_weight+= weight[17];\n        if(cur_weight!=0.0){\n            mean[4] /= cur_weight;\n            sigma[4] /= cur_weight;\n        }\n        cur_std = sigma[4] - mean[4] * mean[4];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[4] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[5]=vec3(0.0);\n        sigma[5]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[5]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * weight[0];\n        sigma[5]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * texture2D(texture, (fc + offset[0]) * Frag).rgb * weight[0];\n        cur_weight+= weight[0];\n        mean[5]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * weight[7];\n        sigma[5]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * texture2D(texture, (fc + offset[7]) * Frag).rgb * weight[7];\n        cur_weight+= weight[7];\n        mean[5]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * weight[14];\n        sigma[5]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * texture2D(texture, (fc + offset[14]) * Frag).rgb * weight[14];\n        cur_weight+= weight[14];\n        mean[5]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * weight[8];\n        sigma[5]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * texture2D(texture, (fc + offset[8]) * Frag).rgb * weight[8];\n        cur_weight+= weight[8];\n        mean[5]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * weight[15];\n        sigma[5]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * texture2D(texture, (fc + offset[15]) * Frag).rgb * weight[15];\n        cur_weight+= weight[15];\n        mean[5]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * weight[16];\n        sigma[5]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * texture2D(texture, (fc + offset[16]) * Frag).rgb * weight[16];\n        cur_weight+= weight[16];\n        if(cur_weight!=0.0){\n            mean[5] /= cur_weight;\n            sigma[5] /= cur_weight;\n        }\n        cur_std = sigma[5] - mean[5] * mean[5];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[5] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[6]=vec3(0.0);\n        sigma[6]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[6]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * weight[21];\n        sigma[6]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2D(texture, (fc + offset[21]) * Frag).rgb * weight[21];\n        cur_weight+= weight[21];\n        mean[6]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * weight[28];\n        sigma[6]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * texture2D(texture, (fc + offset[28]) * Frag).rgb * weight[28];\n        cur_weight+= weight[28];\n        mean[6]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * weight[35];\n        sigma[6]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * texture2D(texture, (fc + offset[35]) * Frag).rgb * weight[35];\n        cur_weight+= weight[35];\n        mean[6]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * weight[22];\n        sigma[6]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2D(texture, (fc + offset[22]) * Frag).rgb * weight[22];\n        cur_weight+= weight[22];\n        mean[6]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * weight[29];\n        sigma[6]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * texture2D(texture, (fc + offset[29]) * Frag).rgb * weight[29];\n        cur_weight+= weight[29];\n        mean[6]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * weight[23];\n        sigma[6]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2D(texture, (fc + offset[23]) * Frag).rgb * weight[23];\n        cur_weight+= weight[23];\n        if(cur_weight!=0.0){\n            mean[6] /= cur_weight;\n            sigma[6] /= cur_weight;\n        }\n        cur_std = sigma[6] - mean[6] * mean[6];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n        total_ms += mean[6] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n        mean[7]=vec3(0.0);\n        sigma[7]=vec3(0.0);\n        cur_weight = 0.0;\n        mean[7]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * weight[42];\n        sigma[7]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * texture2D(texture, (fc + offset[42]) * Frag).rgb * weight[42];\n        cur_weight+= weight[42];\n        mean[7]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * weight[36];\n        sigma[7]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * texture2D(texture, (fc + offset[36]) * Frag).rgb * weight[36];\n        cur_weight+= weight[36];\n        mean[7]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * weight[43];\n        sigma[7]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * texture2D(texture, (fc + offset[43]) * Frag).rgb * weight[43];\n        cur_weight+= weight[43];\n        mean[7]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * weight[30];\n        sigma[7]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * texture2D(texture, (fc + offset[30]) * Frag).rgb * weight[30];\n        cur_weight+= weight[30];\n        mean[7]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * weight[37];\n        sigma[7]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * texture2D(texture, (fc + offset[37]) * Frag).rgb * weight[37];\n        cur_weight+= weight[37];\n        mean[7]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * weight[44];\n        sigma[7]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * texture2D(texture, (fc + offset[44]) * Frag).rgb * weight[44];\n        cur_weight+= weight[44];\n        if(cur_weight!=0.0){\n            mean[7] /= cur_weight;\n            sigma[7] /= cur_weight;\n        }\n        cur_std = sigma[7] - mean[7] * mean[7];\n        if(cur_std.r > 1e-10 && cur_std.g > 1e-10 && cur_std.b > 1e-10){\n            cur_std = sqrt(cur_std);\n        }else{\n            cur_std = vec3(1e-10);\n        }\n\n        total_ms += mean[7] * pow(cur_std,vec3(-q));\n        total_s  += pow(cur_std,vec3(-q));\n\n        if(total_s.r> 1e-10 && total_s.g> 1e-10 && total_s.b> 1e-10){\n            destColor = (total_ms/total_s).rgb;\n            destColor = max(destColor, 0.0);\n            destColor = min(destColor, 1.0);\n        }else{\n            destColor = texture2D(texture, vTexCoord).rgb;\n        }\n\n    }else{\n        destColor = texture2D(texture, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "gkuwaharaFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "grayScaleFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nuniform bool      grayScale;\nvarying vec2      vTexCoord;\n\nconst float redScale   = 0.298912;\nconst float greenScale = 0.586611;\nconst float blueScale  = 0.114478;\nconst vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\nvoid main(void){\n	vec4 smpColor = texture2D(texture, vTexCoord);\n	if(grayScale){\n		float grayColor = dot(smpColor.rgb, monochromeScale);\n		smpColor = vec4(vec3(grayColor), 1.0);\n	}\n	gl_FragColor = smpColor;\n}\n",
    "grayScaleFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "kuwaharaFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\n\nuniform bool b_kuwahara;\nuniform float cvsHeight;\nuniform float cvsWidth;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n    vec3  destColor = vec3(0.0);\n    if(b_kuwahara){\n        float minVal =0.0;\n        vec3 mean[4];\n        vec3 sigma[4];\n        vec2 offset[49];\n        offset[0] = vec2(-3.0, -3.0);\n        offset[1] = vec2(-2.0, -3.0);\n        offset[2] = vec2(-1.0, -3.0);\n        offset[3] = vec2( 0.0, -3.0);\n        offset[4] = vec2( 1.0, -3.0);\n        offset[5] = vec2( 2.0, -3.0);\n        offset[6] = vec2( 3.0, -3.0);\n\n        offset[7]  = vec2(-3.0, -2.0);\n        offset[8]  = vec2(-2.0, -2.0);\n        offset[9]  = vec2(-1.0, -2.0);\n        offset[10] = vec2( 0.0, -2.0);\n        offset[11] = vec2( 1.0, -2.0);\n        offset[12] = vec2( 2.0, -2.0);\n        offset[13] = vec2( 3.0, -2.0);\n\n        offset[14] = vec2(-3.0, -1.0);\n        offset[15] = vec2(-2.0, -1.0);\n        offset[16] = vec2(-1.0, -1.0);\n        offset[17] = vec2( 0.0, -1.0);\n        offset[18] = vec2( 1.0, -1.0);\n        offset[19] = vec2( 2.0, -1.0);\n        offset[20] = vec2( 3.0, -1.0);\n\n        offset[21] = vec2(-3.0,  0.0);\n        offset[22] = vec2(-2.0,  0.0);\n        offset[23] = vec2(-1.0,  0.0);\n        offset[24] = vec2( 0.0,  0.0);\n        offset[25] = vec2( 1.0,  0.0);\n        offset[26] = vec2( 2.0,  0.0);\n        offset[27] = vec2( 3.0,  0.0);\n\n        offset[28] = vec2(-3.0,  1.0);\n        offset[29] = vec2(-2.0,  1.0);\n        offset[30] = vec2(-1.0,  1.0);\n        offset[31] = vec2( 0.0,  1.0);\n        offset[32] = vec2( 1.0,  1.0);\n        offset[33] = vec2( 2.0,  1.0);\n        offset[34] = vec2( 3.0,  1.0);\n\n        offset[35] = vec2(-3.0,  2.0);\n        offset[36] = vec2(-2.0,  2.0);\n        offset[37] = vec2(-1.0,  2.0);\n        offset[38] = vec2( 0.0,  2.0);\n        offset[39] = vec2( 1.0,  2.0);\n        offset[40] = vec2( 2.0,  2.0);\n        offset[41] = vec2( 3.0,  2.0);\n\n        offset[42] = vec2(-3.0,  3.0);\n        offset[43] = vec2(-2.0,  3.0);\n        offset[44] = vec2(-1.0,  3.0);\n        offset[45] = vec2( 0.0,  3.0);\n        offset[46] = vec2( 1.0,  3.0);\n        offset[47] = vec2( 2.0,  3.0);\n        offset[48] = vec2( 3.0,  3.0);\n\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n\n        //calculate mean\n        mean[0] = vec3(0.0);\n        sigma[0] = vec3(0.0);\n        mean[0]  += texture2D(texture, (fc + offset[3]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[4]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[5]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[6]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[10]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[11]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[12]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[13]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[17]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[18]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[19]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[20]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[25]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[26]) * Frag).rgb;\n        mean[0]  += texture2D(texture, (fc + offset[27]) * Frag).rgb;\n\n        sigma[0]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D(texture, (fc + offset[3]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[4]) * Frag).rgb * texture2D(texture, (fc + offset[4]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[5]) * Frag).rgb * texture2D(texture, (fc + offset[5]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[6]) * Frag).rgb * texture2D(texture, (fc + offset[6]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2D(texture, (fc + offset[10]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[11]) * Frag).rgb * texture2D(texture, (fc + offset[11]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[12]) * Frag).rgb * texture2D(texture, (fc + offset[12]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[13]) * Frag).rgb * texture2D(texture, (fc + offset[13]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2D(texture, (fc + offset[17]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[18]) * Frag).rgb * texture2D(texture, (fc + offset[18]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[19]) * Frag).rgb * texture2D(texture, (fc + offset[19]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[20]) * Frag).rgb * texture2D(texture, (fc + offset[20]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2D(texture, (fc + offset[25]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2D(texture, (fc + offset[26]) * Frag).rgb;\n        sigma[0]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2D(texture, (fc + offset[27]) * Frag).rgb;\n\n        mean[0] /= 16.0;\n        sigma[0] = abs(sigma[0]/16.0 -  mean[0]* mean[0]);\n        minVal = sigma[0].r + sigma[0].g + sigma[0].b;\n\n        mean[1] = vec3(0.0);\n        sigma[1] = vec3(0.0);\n        mean[1]  += texture2D(texture, (fc + offset[0]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[1]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[2]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[3]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[7]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[8]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[9]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[10]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[14]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[15]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[16]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[17]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[21]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[22]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[23]) * Frag).rgb;\n        mean[1]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n\n        sigma[1]  += texture2D(texture, (fc + offset[0]) * Frag).rgb * texture2D(texture, (fc + offset[0]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[1]) * Frag).rgb * texture2D(texture, (fc + offset[1]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[2]) * Frag).rgb * texture2D(texture, (fc + offset[2]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[3]) * Frag).rgb * texture2D(texture, (fc + offset[3]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[7]) * Frag).rgb * texture2D(texture, (fc + offset[7]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[8]) * Frag).rgb * texture2D(texture, (fc + offset[8]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[9]) * Frag).rgb * texture2D(texture, (fc + offset[9]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[10]) * Frag).rgb * texture2D(texture, (fc + offset[10]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[14]) * Frag).rgb * texture2D(texture, (fc + offset[14]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[15]) * Frag).rgb * texture2D(texture, (fc + offset[15]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[16]) * Frag).rgb * texture2D(texture, (fc + offset[16]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[17]) * Frag).rgb * texture2D(texture, (fc + offset[17]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2D(texture, (fc + offset[21]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2D(texture, (fc + offset[22]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2D(texture, (fc + offset[23]) * Frag).rgb;\n        sigma[1]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2D(texture, (fc + offset[24]) * Frag).rgb;\n\n        mean[1] /= 16.0;\n        sigma[1] = abs(sigma[1]/16.0 -  mean[1]* mean[1]);\n        float sigmaVal = sigma[1].r + sigma[1].g + sigma[1].b;\n        if(sigmaVal<minVal){\n            destColor = mean[1].rgb;\n            minVal = sigmaVal;\n        }else{\n            destColor = mean[0].rgb;\n        }\n\n        mean[2] = vec3(0.0);\n        sigma[2] = vec3(0.0);\n        mean[2]  += texture2D(texture, (fc + offset[21]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[22]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[23]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[28]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[29]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[30]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[31]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[35]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[36]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[37]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[38]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[42]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[43]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[44]) * Frag).rgb;\n        mean[2]  += texture2D(texture, (fc + offset[45]) * Frag).rgb;\n\n        sigma[2]  += texture2D(texture, (fc + offset[21]) * Frag).rgb * texture2D(texture, (fc + offset[21]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[22]) * Frag).rgb * texture2D(texture, (fc + offset[22]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[23]) * Frag).rgb * texture2D(texture, (fc + offset[23]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[28]) * Frag).rgb * texture2D(texture, (fc + offset[28]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[29]) * Frag).rgb * texture2D(texture, (fc + offset[29]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[30]) * Frag).rgb * texture2D(texture, (fc + offset[30]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2D(texture, (fc + offset[31]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[35]) * Frag).rgb * texture2D(texture, (fc + offset[35]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[36]) * Frag).rgb * texture2D(texture, (fc + offset[36]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[37]) * Frag).rgb * texture2D(texture, (fc + offset[37]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2D(texture, (fc + offset[38]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[42]) * Frag).rgb * texture2D(texture, (fc + offset[42]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[43]) * Frag).rgb * texture2D(texture, (fc + offset[43]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[44]) * Frag).rgb * texture2D(texture, (fc + offset[44]) * Frag).rgb;\n        sigma[2]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2D(texture, (fc + offset[45]) * Frag).rgb;\n\n        mean[2] /= 16.0;\n        sigma[2] = abs(sigma[2]/16.0 -  mean[2]* mean[2]);\n        sigmaVal = sigma[2].r + sigma[2].g + sigma[2].b;\n        if(sigmaVal<minVal){\n            destColor = mean[2].rgb;\n            minVal = sigmaVal;\n        }\n\n        mean[3] = vec3(0.0);\n        sigma[3] = vec3(0.0);\n        mean[3]  += texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[25]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[26]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[27]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[31]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[32]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[33]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[34]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[38]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[39]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[40]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[41]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[45]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[46]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[47]) * Frag).rgb;\n        mean[3]  += texture2D(texture, (fc + offset[48]) * Frag).rgb;\n\n        sigma[3]  += texture2D(texture, (fc + offset[24]) * Frag).rgb * texture2D(texture, (fc + offset[24]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[25]) * Frag).rgb * texture2D(texture, (fc + offset[25]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[26]) * Frag).rgb * texture2D(texture, (fc + offset[26]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[27]) * Frag).rgb * texture2D(texture, (fc + offset[27]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[31]) * Frag).rgb * texture2D(texture, (fc + offset[31]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[32]) * Frag).rgb * texture2D(texture, (fc + offset[32]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[33]) * Frag).rgb * texture2D(texture, (fc + offset[33]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[34]) * Frag).rgb * texture2D(texture, (fc + offset[34]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[38]) * Frag).rgb * texture2D(texture, (fc + offset[38]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[39]) * Frag).rgb * texture2D(texture, (fc + offset[39]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[40]) * Frag).rgb * texture2D(texture, (fc + offset[40]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[41]) * Frag).rgb * texture2D(texture, (fc + offset[41]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[45]) * Frag).rgb * texture2D(texture, (fc + offset[45]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[46]) * Frag).rgb * texture2D(texture, (fc + offset[46]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[47]) * Frag).rgb * texture2D(texture, (fc + offset[47]) * Frag).rgb;\n        sigma[3]  += texture2D(texture, (fc + offset[48]) * Frag).rgb * texture2D(texture, (fc + offset[48]) * Frag).rgb;\n\n        mean[3] /= 16.0;\n        sigma[3] = abs(sigma[3]/16.0 -  mean[3]* mean[3]);\n        sigmaVal = sigma[3].r + sigma[3].g + sigma[3].b;\n        if(sigmaVal<minVal){\n            destColor = mean[3].rgb;\n            minVal = sigmaVal;\n        }  \n\n    }else{\n        destColor = texture2D(texture, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "kuwaharaFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "laplacianFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\n\nuniform bool b_laplacian;\nuniform float cvsHeight;\nuniform float cvsWidth;\nuniform float coef[9];\nvarying vec2 vTexCoord;\n\nconst float redScale   = 0.298912;\nconst float greenScale = 0.586611;\nconst float blueScale  = 0.114478;\nconst vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\nvoid main(void){\n    vec3  destColor = vec3(0.0);\n    if(b_laplacian){\n        vec2 offset[9];\n        offset[0] = vec2(-1.0, -1.0);\n        offset[1] = vec2( 0.0, -1.0);\n        offset[2] = vec2( 1.0, -1.0);\n        offset[3] = vec2(-1.0,  0.0);\n        offset[4] = vec2( 0.0,  0.0);\n        offset[5] = vec2( 1.0,  0.0);\n        offset[6] = vec2(-1.0,  1.0);\n        offset[7] = vec2( 0.0,  1.0);\n        offset[8] = vec2( 1.0,  1.0);\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n\n        destColor  += texture2D(texture, (fc + offset[0]) * Frag).rgb * coef[0];\n        destColor  += texture2D(texture, (fc + offset[1]) * Frag).rgb * coef[1];\n        destColor  += texture2D(texture, (fc + offset[2]) * Frag).rgb * coef[2];\n        destColor  += texture2D(texture, (fc + offset[3]) * Frag).rgb * coef[3];\n        destColor  += texture2D(texture, (fc + offset[4]) * Frag).rgb * coef[4];\n        destColor  += texture2D(texture, (fc + offset[5]) * Frag).rgb * coef[5];\n        destColor  += texture2D(texture, (fc + offset[6]) * Frag).rgb * coef[6];\n        destColor  += texture2D(texture, (fc + offset[7]) * Frag).rgb * coef[7];\n        destColor  += texture2D(texture, (fc + offset[8]) * Frag).rgb * coef[8];\n\n        destColor =max(destColor, 0.0);\n    }else{\n        destColor = texture2D(texture, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "laplacianFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "LIC-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\n\nuniform bool b_lic;\nuniform float cvsHeight;\nuniform float cvsWidth;\nuniform float sigma;\n\nstruct lic_t { \n    vec2 p; \n    vec2 t;\n    float w;\n    float dw;\n};\n\nvoid step(inout lic_t s) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 t = texture2D(tfm, s.p).xy;\n    if (dot(t, s.t) < 0.0) t = -t;\n    s.t = t;\n\n    s.dw = (abs(t.x) > abs(t.y))? \n        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : \n        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);\n\n    s.p += t * s.dw / src_size;\n    s.w += s.dw;\n}\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    float twoSigma2 = 2.0 * sigma * sigma;\n    float halfWidth = 2.0 * sigma;\n    vec2 uv = vec2(gl_FragCoord.x / src_size.x, (src_size.y - gl_FragCoord.y) / src_size.y);\n\n    if(b_lic){\n        const int MAX_NUM_ITERATION = 99999;\n        vec3 c = texture2D( src, uv ).xyz;\n        float w = 1.0;\n\n        lic_t a, b;\n        a.p = b.p = uv;\n        a.t = texture2D( tfm, uv ).xy / src_size;\n        b.t = -a.t;\n        a.w = b.w = 0.0;\n\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (a.w < halfWidth) {\n                step(a);\n                float k = a.dw * exp(-a.w * a.w / twoSigma2);\n                c += k * texture2D(src, a.p).xyz;\n                w += k;\n            }else{\n                break;\n            }\n        }\n\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n            if (b.w < halfWidth) {\n                step(b);\n                float k = b.dw * exp(-b.w * b.w / twoSigma2);\n                c += k * texture2D(src, b.p).xyz;\n                w += k;\n            }else{\n                break;\n            }\n        }\n\n        gl_FragColor = vec4(c / w, 1.0);\n    }else{\n        gl_FragColor = texture2D(src, uv);\n    }\n\n}\n",
    "LIC-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "luminance-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nuniform float threshold;\nvarying vec2 vTexCoord;\n\nconst float redScale   = 0.298912;\nconst float greenScale = 0.586611;\nconst float blueScale  = 0.114478;\nconst vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\nvoid main(void){\n\n	vec4 smpColor = texture2D(texture, vec2(vTexCoord.s, 1.0 - vTexCoord.t));\n	float luminance = dot(smpColor.rgb, monochromeScale);\n	if(luminance<threshold){luminance = 0.0;}\n\n	smpColor = vec4(vec3(luminance), 1.0);\n	gl_FragColor =smpColor;\n}\n",
    "luminance-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "phong-frag":
        "precision mediump float;\n\nuniform mat4 invMatrix;\nuniform vec3 lightDirection;\nuniform vec3 eyeDirection;\nuniform vec4 ambientColor;\nvarying vec4 vColor;\nvarying vec3 vNormal;\n\nvoid main(void){\n	vec3 invLight = normalize(invMatrix*vec4(lightDirection,0.0)).xyz;\n	vec3 invEye = normalize(invMatrix*vec4(eyeDirection,0.0)).xyz;\n	vec3 halfLE = normalize(invLight+invEye);\n	float diffuse = clamp(dot(vNormal,invLight),0.0,1.0);\n	float specular = pow(clamp(dot(vNormal,halfLE),0.0,1.0),50.0);\n	vec4 destColor = vColor * vec4(vec3(diffuse),1.0) + vec4(vec3(specular),1.0) + ambientColor;\n	gl_FragColor = destColor;\n}\n",
    "phong-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec3 normal;\n\nuniform mat4 mvpMatrix;\n\nvarying vec4 vColor;\nvarying vec3 vNormal;\n\nvoid main(void){\n    vNormal = normal;\n    vColor = color;\n    gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "point-frag": "precision mediump float;\nvarying vec4      vColor;\n\nvoid main(void){\n    gl_FragColor = vColor;\n}\n",
    "point-vert":
        "attribute vec3 position;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nuniform   float pointSize;\nvarying   vec4 vColor;\n\nvoid main(void){\n    vColor        = color;\n    gl_Position   = mvpMatrix * vec4(position, 1.0);\n    gl_PointSize  = pointSize;\n}\n",
    "pointLighting-frag":
        "precision mediump float;\n\nuniform mat4 invMatrix;\nuniform vec3 lightPosition;\nuniform vec3 eyeDirection;\nuniform vec4 ambientColor;\n\nvarying vec4 vColor;\nvarying vec3 vNormal;\nvarying vec3 vPosition;\n\nvoid main(void){\n	vec3 lightVec = lightPosition -vPosition;\n	vec3 invLight = normalize(invMatrix*vec4(lightVec,0.0)).xyz;\n	vec3 invEye = normalize(invMatrix*vec4(eyeDirection,0.0)).xyz;\n	vec3 halfLE = normalize(invLight+invEye);\n	float diffuse = clamp(dot(vNormal,invLight),0.0,1.0);\n	float specular = pow(clamp(dot(vNormal,halfLE),0.0,1.0),50.0);\n	vec4 destColor = vColor * vec4(vec3(diffuse),1.0) + vec4(vec3(specular),1.0) + ambientColor;\n	gl_FragColor = destColor;\n}\n",
    "pointLighting-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec3 normal;\n\nuniform mat4 mvpMatrix;\nuniform mat4 mMatrix;\n\nvarying vec3 vPosition;\nvarying vec4 vColor;\nvarying vec3 vNormal;\n\nvoid main(void){\n    vPosition = (mMatrix*vec4(position,1.0)).xyz;\n    vNormal = normal;\n    vColor = color;\n    gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "pointSprite-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nvarying vec4      vColor;\n\nvoid main(void){\n    vec4 smpColor = vec4(1.0);\n    smpColor = texture2D(texture,gl_PointCoord);\n    if(smpColor.a == 0.0){\n        discard;\n    }else{\n        gl_FragColor = vColor * smpColor;\n    }\n}\n",
    "pointSprite-vert":
        "attribute vec3 position;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nuniform   float pointSize;\nvarying   vec4 vColor;\n\nvoid main(void){\n    vColor        = color;\n    gl_Position   = mvpMatrix * vec4(position, 1.0);\n    gl_PointSize  = pointSize;\n}\n",
    "projTexture-frag":
        "precision mediump float;\n\nuniform mat4      invMatrix;\nuniform vec3      lightPosition;\nuniform sampler2D texture;\nvarying vec3      vPosition;\nvarying vec3      vNormal;\nvarying vec4      vColor;\nvarying vec4      vTexCoord;\n\nvoid main(void){\n	vec3  light    = lightPosition - vPosition;\n	vec3  invLight = normalize(invMatrix * vec4(light, 0.0)).xyz;\n	float diffuse  = clamp(dot(vNormal, invLight), 0.1, 1.0);\n	vec4  smpColor = texture2DProj(texture, vTexCoord);\n	gl_FragColor   = vColor * (0.5 + diffuse) * smpColor;\n}\n",
    "projTexture-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mMatrix;\nuniform   mat4 tMatrix;\nuniform   mat4 mvpMatrix;\nvarying   vec3 vPosition;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\nvarying   vec4 vTexCoord;\n\nvoid main(void){\n	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n	vNormal     = normal;\n	vColor      = color;\n	vTexCoord   = tMatrix * vec4(vPosition, 1.0);\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "P_FDoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\n\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma_e;\nuniform float sigma_r;\nuniform float tau;\n\nuniform bool b_FDoG;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n\n    vec3 destColor = vec3(0.0);\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 uv = gl_FragCoord.xy /src_size;\n    if(b_FDoG){\n        float twoSigmaESquared = 2.0 * sigma_e * sigma_e;\n        float twoSigmaRSquared = 2.0 * sigma_r * sigma_r;\n\n        vec2 t = texture2D(tfm, uv).xy;\n        vec2 n = vec2(t.y, -t.x);\n        vec2 nabs = abs(n);\n        float ds = 1.0 / ((nabs.x > nabs.y)? nabs.x : nabs.y);\n        n /= src_size;\n\n        vec2 sum = texture2D( src, uv ).xx;\n        vec2 norm = vec2(1.0, 1.0);\n\n        float halfWidth = 2.0 * sigma_r;\n        float d = ds;\n        const int MAX_NUM_ITERATION = 99999;\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n\n            if( d <= halfWidth) {\n                vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n                                    exp( -d * d / twoSigmaRSquared ));\n                norm += 2.0 * kernel;\n\n                vec2 L0 = texture2D( src, uv - d*n ).xx;\n                vec2 L1 = texture2D( src, uv + d*n ).xx;\n                sum += kernel * ( L0 + L1 );\n            }else{\n                break;\n            }\n            d+=ds;\n        }\n\n        sum /= norm;\n\n        float diff = 100.0 * (sum.x - tau * sum.y);\n        destColor= vec3(diff);\n    }else{\n        destColor = texture2D(src, uv).rgb;\n    }\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "P_FDoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "P_FXDoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\nuniform sampler2D tfm;\n\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma;\nuniform float k;\nuniform float p;\n\nuniform bool b_FXDoG;\nvarying vec2 vTexCoord;\n\nvoid main(void){\n\n    vec3 destColor = vec3(0.0);\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 uv = gl_FragCoord.xy /src_size;\n    if(b_FXDoG){\n        float twoSigmaESquared = 2.0 * sigma * sigma;\n        float twoSigmaRSquared = twoSigmaESquared * k * k;\n\n        vec2 t = texture2D(tfm, uv).xy;\n        vec2 n = vec2(t.y, -t.x);\n        vec2 nabs = abs(n);\n        float ds = 1.0 / ((nabs.x > nabs.y)? nabs.x : nabs.y);\n        n /= src_size;\n\n        vec2 sum = texture2D( src, uv ).xx;\n        vec2 norm = vec2(1.0, 1.0);\n\n        float halfWidth = 2.0 * sigma;\n        float d = ds;\n        const int MAX_NUM_ITERATION = 99999;\n        for(int i = 0;i<MAX_NUM_ITERATION ;i++){\n\n            if( d <= halfWidth) {\n                vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n                                    exp( -d * d / twoSigmaRSquared ));\n                norm += 2.0 * kernel;\n\n                vec2 L0 = texture2D( src, uv - d*n ).xx;\n                vec2 L1 = texture2D( src, uv + d*n ).xx;\n                sum += kernel * ( L0 + L1 );\n            }else{\n                break;\n            }\n            d+=ds;\n        }\n\n        sum /= norm;\n\n        float diff = 100.0 * ((1.0 + p) * sum.x - p * sum.y);\n        destColor= vec3(diff);\n    }else{\n        destColor = texture2D(src, uv).rgb;\n    }\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "P_FXDoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "refractionMapping-frag":
        "precision mediump float;\n\nuniform vec3        eyePosition;\nuniform samplerCube cubeTexture;\nuniform bool        refraction;\nvarying vec3        vPosition;\nvarying vec3        vNormal;\nvarying vec4        vColor;\n\n//reflact calculation TODO\n//vec3 egt_refract(vec3 p, vec3 n,float eta){\n//}\n\nvoid main(void){\n	vec3 ref;\n	if(refraction){\n		ref = refract(normalize(vPosition - eyePosition), vNormal,0.6);\n	}else{\n		ref = vNormal;\n	}\n	vec4 envColor  = textureCube(cubeTexture, ref);\n	vec4 destColor = vColor * envColor;\n	gl_FragColor   = destColor;\n}\n",
    "refractionMapping-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nvarying   vec3 vPosition;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n	vNormal     = normalize((mMatrix * vec4(normal, 0.0)).xyz);\n	vColor      = color;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "sepiaFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nuniform bool      sepia;\nvarying vec2      vTexCoord;\n\nconst float redScale   = 0.298912;\nconst float greenScale = 0.586611;\nconst float blueScale  = 0.114478;\nconst vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\nconst float sRedScale   = 1.07;\nconst float sGreenScale = 0.74;\nconst float sBlueScale  = 0.43;\nconst vec3  sepiaScale = vec3(sRedScale, sGreenScale, sBlueScale);\n\nvoid main(void){\n    vec4  smpColor  = texture2D(texture, vTexCoord);\n    float grayColor = dot(smpColor.rgb, monochromeScale);\n\n    vec3 monoColor = vec3(grayColor) * sepiaScale; \n    smpColor = vec4(monoColor, 1.0);\n\n    gl_FragColor = smpColor;\n}\n",
    "sepiaFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "shadowDepthBuffer-frag":
        "precision mediump float;\n\nuniform bool depthBuffer;\n\nvarying vec4 vPosition;\n\nvec4 convRGBA(float depth){\n    float r = depth;\n    float g = fract(r*255.0);\n    float b = fract(g*255.0); \n    float a = fract(b*255.0);\n    float coef = 1.0/255.0;\n    r-= g* coef; \n    g-= b* coef; \n    b-= a* coef; \n    return vec4(r,g,b,a);\n}\n\nvoid main(void){\n    vec4 convColor;\n    if(depthBuffer){\n        convColor = convRGBA(gl_FragCoord.z);\n    }else{\n        float near = 0.1;\n        float far  = 150.0;\n        float linerDepth = 1.0 / (far - near);\n        linerDepth *= length(vPosition);\n        convColor = convRGBA(linerDepth);\n    }\n    gl_FragColor = convColor;\n}\n",
    "shadowDepthBuffer-vert": "attribute vec3 position;\nuniform mat4 mvpMatrix;\n\nvarying vec4 vPosition;\n\nvoid main(void){\n    vPosition = mvpMatrix * vec4(position, 1.0);\n    gl_Position = vPosition;\n}\n",
    "shadowScreen-frag":
        "precision mediump float;\n\nuniform mat4      invMatrix;\nuniform vec3      lightPosition;\nuniform sampler2D texture;\nuniform bool      depthBuffer;\nvarying vec3      vPosition;\nvarying vec3      vNormal;\nvarying vec4      vColor;\nvarying vec4      vTexCoord;\nvarying vec4      vDepth;\n\nfloat restDepth(vec4 RGBA){\n    const float rMask = 1.0;\n    const float gMask = 1.0 / 255.0;\n    const float bMask = 1.0 / (255.0 * 255.0);\n    const float aMask = 1.0 / (255.0 * 255.0 * 255.0);\n    float depth = dot(RGBA, vec4(rMask, gMask, bMask, aMask));\n    return depth;\n}\n\nvoid main(void){\n    vec3  light     = lightPosition - vPosition;\n    vec3  invLight  = normalize(invMatrix * vec4(light, 0.0)).xyz;\n    float diffuse   = clamp(dot(vNormal, invLight), 0.1, 1.0);\n    float shadow    = restDepth(texture2DProj(texture, vTexCoord));\n    vec4 depthColor = vec4(1.0);\n    if(vDepth.w > 0.0){\n        if(depthBuffer){\n            vec4 lightCoord = vDepth / vDepth.w;\n            if(lightCoord.z - 0.0001 > shadow){\n                depthColor  = vec4(0.5, 0.5, 0.5, 1.0);\n            }\n        }else{\n            float near = 0.1;\n            float far  = 150.0;\n            float linerDepth = 1.0 / (far - near);\n            linerDepth *= length(vPosition.xyz - lightPosition);\n            if(linerDepth - 0.0001 > shadow){\n                depthColor  = vec4(0.5, 0.5, 0.5, 1.0);\n            }\n        }\n    }\n    gl_FragColor = vColor * (vec3(diffuse),1.0) * depthColor;\n}\n",
    "shadowScreen-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mMatrix;\nuniform   mat4 mvpMatrix;\nuniform   mat4 tMatrix;\nuniform   mat4 lgtMatrix;\nvarying   vec3 vPosition;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\nvarying   vec4 vTexCoord;\nvarying   vec4 vDepth;\n\nvoid main(void){\n    vPosition   = (mMatrix * vec4(position, 1.0)).xyz;\n    vNormal     = normal;\n    vColor      = color;\n    vTexCoord   = tMatrix * vec4(vPosition, 1.0);\n    vDepth      = lgtMatrix * vec4(position, 1.0);\n    gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "sobelFilter-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\n\nuniform bool b_sobel;\nuniform float cvsHeight;\nuniform float cvsWidth;\nuniform float hCoef[9];\nuniform float vCoef[9];\nvarying vec2 vTexCoord;\n\nvoid main(void){\n    vec3 destColor = vec3(0.0);\n    if(b_sobel){\n        vec2 offset[9];\n        offset[0] = vec2(-1.0, -1.0);\n        offset[1] = vec2( 0.0, -1.0);\n        offset[2] = vec2( 1.0, -1.0);\n        offset[3] = vec2(-1.0,  0.0);\n        offset[4] = vec2( 0.0,  0.0);\n        offset[5] = vec2( 1.0,  0.0);\n        offset[6] = vec2(-1.0,  1.0);\n        offset[7] = vec2( 0.0,  1.0);\n        offset[8] = vec2( 1.0,  1.0);\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2  fc = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n        vec3  horizonColor = vec3(0.0);\n        vec3  verticalColor = vec3(0.0);\n\n        horizonColor  += texture2D(texture, (fc + offset[0]) * Frag).rgb * hCoef[0];\n        horizonColor  += texture2D(texture, (fc + offset[1]) * Frag).rgb * hCoef[1];\n        horizonColor  += texture2D(texture, (fc + offset[2]) * Frag).rgb * hCoef[2];\n        horizonColor  += texture2D(texture, (fc + offset[3]) * Frag).rgb * hCoef[3];\n        horizonColor  += texture2D(texture, (fc + offset[4]) * Frag).rgb * hCoef[4];\n        horizonColor  += texture2D(texture, (fc + offset[5]) * Frag).rgb * hCoef[5];\n        horizonColor  += texture2D(texture, (fc + offset[6]) * Frag).rgb * hCoef[6];\n        horizonColor  += texture2D(texture, (fc + offset[7]) * Frag).rgb * hCoef[7];\n        horizonColor  += texture2D(texture, (fc + offset[8]) * Frag).rgb * hCoef[8];\n\n        verticalColor += texture2D(texture, (fc + offset[0]) * Frag).rgb * vCoef[0];\n        verticalColor += texture2D(texture, (fc + offset[1]) * Frag).rgb * vCoef[1];\n        verticalColor += texture2D(texture, (fc + offset[2]) * Frag).rgb * vCoef[2];\n        verticalColor += texture2D(texture, (fc + offset[3]) * Frag).rgb * vCoef[3];\n        verticalColor += texture2D(texture, (fc + offset[4]) * Frag).rgb * vCoef[4];\n        verticalColor += texture2D(texture, (fc + offset[5]) * Frag).rgb * vCoef[5];\n        verticalColor += texture2D(texture, (fc + offset[6]) * Frag).rgb * vCoef[6];\n        verticalColor += texture2D(texture, (fc + offset[7]) * Frag).rgb * vCoef[7];\n        verticalColor += texture2D(texture, (fc + offset[8]) * Frag).rgb * vCoef[8];\n        destColor = vec3(sqrt(horizonColor * horizonColor + verticalColor * verticalColor));\n    }else{\n        destColor = texture2D(texture, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "sobelFilter-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "specCpt-frag": "precision mediump float;\n\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "specCpt-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nuniform   mat4 invMatrix;\nuniform   vec3 lightDirection;\nuniform   vec3 eyeDirection;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n	vec3  invEye   = normalize(invMatrix * vec4(eyeDirection, 0.0)).xyz;\n	vec3  halfLE   = normalize(invLight + invEye);\n	float specular = pow(clamp(dot(normal, halfLE), 0.0, 1.0), 50.0);\n	vColor         = color * vec4(vec3(specular), 1.0);\n	gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "specular-frag": "precision mediump float;\n\nvarying vec4 vColor;\n\nvoid main(void){\n	gl_FragColor = vColor;\n}\n",
    "specular-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec3 normal;\n\nuniform mat4 mvpMatrix;\nuniform mat4 invMatrix;\n\nuniform vec3 lightDirection;\nuniform vec3 eyeDirection;\nuniform vec4 ambientColor;\nvarying vec4 vColor;\n\nvoid main(void){\n    vec3 invLight = normalize(invMatrix*vec4(lightDirection,0.0)).xyz;\n    vec3 invEye = normalize(invMatrix* vec4(eyeDirection,0.0)).xyz;\n    vec3 halfLE = normalize(invLight+invEye);\n\n    float diffuse = clamp(dot(invLight,normal),0.0,1.0);\n    float specular = pow(clamp(dot(normal,halfLE),0.0,1.0),50.0);\n    vec4 light = color*vec4(vec3(diffuse),1.0)+vec4(vec3(specular),1.0);\n    vColor = light + ambientColor;\n    gl_Position    = mvpMatrix * vec4(position, 1.0);\n}\n",
    "SST-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nvoid main (void) {\n    vec2 src_size = vec2(cvsWidth, cvsHeight);\n    vec2 uv = gl_FragCoord.xy / src_size;\n    vec2 d = 1.0 / src_size;\n    vec3 u = (\n        -1.0 * texture2D(src, uv + vec2(-d.x, -d.y)).xyz +\n        -2.0 * texture2D(src, uv + vec2(-d.x,  0.0)).xyz + \n        -1.0 * texture2D(src, uv + vec2(-d.x,  d.y)).xyz +\n        +1.0 * texture2D(src, uv + vec2( d.x, -d.y)).xyz +\n        +2.0 * texture2D(src, uv + vec2( d.x,  0.0)).xyz + \n        +1.0 * texture2D(src, uv + vec2( d.x,  d.y)).xyz\n        ) / 4.0;\n\n    vec3 v = (\n           -1.0 * texture2D(src, uv + vec2(-d.x, -d.y)).xyz + \n           -2.0 * texture2D(src, uv + vec2( 0.0, -d.y)).xyz + \n           -1.0 * texture2D(src, uv + vec2( d.x, -d.y)).xyz +\n           +1.0 * texture2D(src, uv + vec2(-d.x,  d.y)).xyz +\n           +2.0 * texture2D(src, uv + vec2( 0.0,  d.y)).xyz + \n           +1.0 * texture2D(src, uv + vec2( d.x,  d.y)).xyz\n           ) / 4.0;\n\n    gl_FragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);\n}\n",
    "SST-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "stencilBufferOutline-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nuniform bool      useTexture;\nvarying vec4      vColor;\nvarying vec2      vTextureCoord;\n\nvoid main(void){\n	vec4 smpColor = vec4(1.0);\n	if(useTexture){\n		smpColor = texture2D(texture, vTextureCoord);\n	}\n	gl_FragColor = vColor * smpColor;\n}\n",
    "stencilBufferOutline-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nattribute vec2 textureCoord;\nuniform   mat4 mvpMatrix;\nuniform   mat4 invMatrix;\nuniform   vec3 lightDirection;\nuniform   bool useLight;\nuniform   bool outline;\nvarying   vec4 vColor;\nvarying   vec2 vTextureCoord;\n\nvoid main(void){\n	if(useLight){\n		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n		float diffuse  = clamp(dot(normal, invLight), 0.1, 1.0);\n		vColor         = color * vec4(vec3(diffuse), 1.0);\n	}else{\n		vColor         = color;\n	}\n	vTextureCoord      = textureCoord;\n	vec3 oPosition     = position;\n	if(outline){\n		oPosition     += normal * 0.1;\n	}\n	gl_Position = mvpMatrix * vec4(oPosition, 1.0);\n}\n",
    "synth-frag":
        "precision mediump float;\n\nuniform sampler2D texture1;\nuniform sampler2D texture2;\nuniform bool      glare;\nvarying vec2      vTexCoord;\n\nvoid main(void){\n	vec4  destColor = texture2D(texture1, vTexCoord);\n	vec4  smpColor  = texture2D(texture2, vec2(vTexCoord.s, 1.0 - vTexCoord.t));\n	if(glare){\n		destColor += smpColor * 0.4;\n	}\n	gl_FragColor = destColor;\n}\n",
    "synth-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "texture-frag":
        "precision mediump float;\n\nuniform sampler2D texture;\nvarying vec4      vColor;\nvarying vec2      vTextureCoord;\n\nvoid main(void){\n    vec4 smpColor = texture2D(texture, vTextureCoord);\n    gl_FragColor  = vColor * smpColor;\n}\n",
    "texture-vert":
        "attribute vec3 position;\nattribute vec4 color;\nattribute vec2 textureCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec4 vColor;\nvarying   vec2 vTextureCoord;\n\nvoid main(void){\n    vColor        = color;\n    vTextureCoord = textureCoord;\n    gl_Position   = mvpMatrix * vec4(position, 1.0);\n}\n",
    "TF-frag":
        "// Tangent Field\nprecision mediump float;\n\nuniform sampler2D src;\nuniform float cvsHeight;\nuniform float cvsWidth;\nuniform float hCoef[9];\nuniform float vCoef[9];\n\nconst float redScale   = 0.298912;\nconst float greenScale = 0.586611;\nconst float blueScale  = 0.114478;\nconst vec3  monochromeScale = vec3(redScale, greenScale, blueScale);\n\nvoid main (void) {\n    vec2 offset[9];\n    offset[0] = vec2(-1.0, -1.0);\n    offset[1] = vec2( 0.0, -1.0);\n    offset[2] = vec2( 1.0, -1.0);\n    offset[3] = vec2(-1.0,  0.0);\n    offset[4] = vec2( 0.0,  0.0);\n    offset[5] = vec2( 1.0,  0.0);\n    offset[6] = vec2(-1.0,  1.0);\n    offset[7] = vec2( 0.0,  1.0);\n    offset[8] = vec2( 1.0,  1.0);\n    float tFrag = 1.0 / cvsHeight;\n    float sFrag = 1.0 / cvsWidth;\n    vec2  Frag = vec2(sFrag,tFrag);\n    vec2  uv = vec2(gl_FragCoord.s, gl_FragCoord.t);\n    float  horizonColor = 0.0;\n    float  verticalColor = 0.0;\n\n    horizonColor  += dot(texture2D(src, (uv + offset[0]) * Frag).rgb, monochromeScale) * hCoef[0];\n    horizonColor  += dot(texture2D(src, (uv + offset[1]) * Frag).rgb, monochromeScale) * hCoef[1];\n    horizonColor  += dot(texture2D(src, (uv + offset[2]) * Frag).rgb, monochromeScale) * hCoef[2];\n    horizonColor  += dot(texture2D(src, (uv + offset[3]) * Frag).rgb, monochromeScale) * hCoef[3];\n    horizonColor  += dot(texture2D(src, (uv + offset[4]) * Frag).rgb, monochromeScale) * hCoef[4];\n    horizonColor  += dot(texture2D(src, (uv + offset[5]) * Frag).rgb, monochromeScale) * hCoef[5];\n    horizonColor  += dot(texture2D(src, (uv + offset[6]) * Frag).rgb, monochromeScale) * hCoef[6];\n    horizonColor  += dot(texture2D(src, (uv + offset[7]) * Frag).rgb, monochromeScale) * hCoef[7];\n    horizonColor  += dot(texture2D(src, (uv + offset[8]) * Frag).rgb, monochromeScale) * hCoef[8];\n\n    verticalColor += dot(texture2D(src, (uv + offset[0]) * Frag).rgb, monochromeScale) * vCoef[0];\n    verticalColor += dot(texture2D(src, (uv + offset[1]) * Frag).rgb, monochromeScale) * vCoef[1];\n    verticalColor += dot(texture2D(src, (uv + offset[2]) * Frag).rgb, monochromeScale) * vCoef[2];\n    verticalColor += dot(texture2D(src, (uv + offset[3]) * Frag).rgb, monochromeScale) * vCoef[3];\n    verticalColor += dot(texture2D(src, (uv + offset[4]) * Frag).rgb, monochromeScale) * vCoef[4];\n    verticalColor += dot(texture2D(src, (uv + offset[5]) * Frag).rgb, monochromeScale) * vCoef[5];\n    verticalColor += dot(texture2D(src, (uv + offset[6]) * Frag).rgb, monochromeScale) * vCoef[6];\n    verticalColor += dot(texture2D(src, (uv + offset[7]) * Frag).rgb, monochromeScale) * vCoef[7];\n    verticalColor += dot(texture2D(src, (uv + offset[8]) * Frag).rgb, monochromeScale) * vCoef[8];\n\n    float mag = sqrt(horizonColor * horizonColor + verticalColor * verticalColor);\n    float vx = verticalColor/mag;\n    float vy = horizonColor/mag;\n\n    gl_FragColor = vec4(vx, vy, mag, 1.0);\n}\n",
    "TF-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "TFM-frag":
        "// by Jan Eric Kyprianidis <www.kyprianidis.com>\nprecision mediump float;\n\nuniform sampler2D src;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nvoid main (void) {\n    vec2 uv = gl_FragCoord.xy / vec2(cvsWidth, cvsHeight);\n    vec3 g = texture2D(src, uv).xyz;\n\n    float lambda1 = 0.5 * (g.y + g.x + sqrt(g.y*g.y - 2.0*g.x*g.y + g.x*g.x + 4.0*g.z*g.z));\n    float lambda2 = 0.5 * (g.y + g.x - sqrt(g.y*g.y - 2.0*g.x*g.y + g.x*g.x + 4.0*g.z*g.z));\n\n    vec2 v = vec2(lambda1 - g.x, -g.z);\n    vec2 t;\n    if (length(v) > 0.0) { \n        t = normalize(v);\n    } else {\n        t = vec2(0.0, 1.0);\n    }\n\n    float phi = atan(t.y, t.x);\n\n    float A = (lambda1 + lambda2 > 0.0)?(lambda1 - lambda2) / (lambda1 + lambda2) : 0.0;\n    gl_FragColor = vec4(t, phi, A);\n}\n",
    "TFM-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
    "toonShading-frag":
        "precision mediump float;\n\nuniform mat4      invMatrix;\nuniform vec3      lightDirection;\nuniform sampler2D texture;\nuniform vec4      edgeColor;\nvarying vec3      vNormal;\nvarying vec4      vColor;\n\nvoid main(void){\n	if(edgeColor.a > 0.0){\n		gl_FragColor   = edgeColor;\n	}else{\n		vec3  invLight = normalize(invMatrix * vec4(lightDirection, 0.0)).xyz;\n		float diffuse  = clamp(dot(vNormal, invLight), 0.1, 1.0);\n		vec4  smpColor = texture2D(texture, vec2(diffuse, 0.0));\n		gl_FragColor   = vColor * smpColor;\n	}\n}\n",
    "toonShading-vert":
        "attribute vec3 position;\nattribute vec3 normal;\nattribute vec4 color;\nuniform   mat4 mvpMatrix;\nuniform   bool edge;\nvarying   vec3 vNormal;\nvarying   vec4 vColor;\n\nvoid main(void){\n	vec3 pos    = position;\n	if(edge){\n		pos    += normal * 0.05;\n	}\n	vNormal     = normal;\n	vColor      = color;\n	gl_Position = mvpMatrix * vec4(pos, 1.0);\n}\n",
    "XDoG-frag":
        "precision mediump float;\n\nuniform sampler2D src;\n\nuniform bool b_XDoG;\nuniform float cvsHeight;\nuniform float cvsWidth;\n\nuniform float sigma;\nuniform float k;\nuniform float p;\nuniform float epsilon;\nuniform float phi;\nvarying vec2 vTexCoord;\n\nfloat cosh(float val)\n{\n    float tmp = exp(val);\n    float cosH = (tmp + 1.0 / tmp) / 2.0;\n    return cosH;\n}\n\nfloat tanh(float val)\n{\n    float tmp = exp(val);\n    float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n    return tanH;\n}\n\nfloat sinh(float val)\n{\n    float tmp = exp(val);\n    float sinH = (tmp - 1.0 / tmp) / 2.0;\n    return sinH;\n}\n\nvoid main(void){\n    vec3 destColor = vec3(0.0);\n    if(b_XDoG){\n        float tFrag = 1.0 / cvsHeight;\n        float sFrag = 1.0 / cvsWidth;\n        vec2  Frag = vec2(sFrag,tFrag);\n        vec2 uv = vec2(gl_FragCoord.s, cvsHeight - gl_FragCoord.t);\n        float twoSigmaESquared = 2.0 * sigma * sigma;\n        float twoSigmaRSquared = twoSigmaESquared * k * k;\n        int halfWidth = int(ceil( 1.0 * sigma * k ));\n\n        const int MAX_NUM_ITERATION = 99999;\n        vec2 sum = vec2(0.0);\n        vec2 norm = vec2(0.0);\n\n        for(int cnt=0;cnt<MAX_NUM_ITERATION;cnt++){\n            if(cnt > (2*halfWidth+1)*(2*halfWidth+1)){break;}\n            int i = int(cnt / (2*halfWidth+1)) - halfWidth;\n            int j = cnt - halfWidth - int(cnt / (2*halfWidth+1)) * (2*halfWidth+1);\n\n            float d = length(vec2(i,j));\n            vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n                                exp( -d * d / twoSigmaRSquared ));\n\n            vec2 L = texture2D(src, (uv + vec2(i,j)) * Frag).xx;\n\n            norm += kernel;\n            sum += kernel * L;\n        }\n\n        sum /= norm;\n\n        float H = 100.0 * ((1.0 + p) * sum.x - p * sum.y);\n        float edge = ( H > epsilon )? 1.0 : 1.0 + tanh( phi * (H - epsilon));\n        destColor = vec3(edge);\n    }else{\n        destColor = texture2D(src, vTexCoord).rgb;\n    }\n\n    gl_FragColor = vec4(destColor, 1.0);\n}\n",
    "XDoG-vert": "attribute vec3 position;\nattribute vec2 texCoord;\nuniform   mat4 mvpMatrix;\nvarying   vec2 vTexCoord;\n\nvoid main(void){\n	vTexCoord   = texCoord;\n	gl_Position = mvpMatrix * vec4(position, 1.0);\n}\n",
};
var EcognitaMathLib;
(function (c) {
    var e = (function () {
        function f(m, q, n, k, p) {
            if (m === void 0) {
                m = undefined;
            }
            if (q === void 0) {
                q = undefined;
            }
            if (n === void 0) {
                n = true;
            }
            if (k === void 0) {
                k = true;
            }
            if (p === void 0) {
                p = false;
            }
            this.data = new Array();
            var l = [-1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 0, 1];
            var o = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0];
            this.index = [0, 1, 2, 3, 2, 1];
            var g = [0, 0, 1, 0, 0, 1, 1, 1];
            for (var j = 0; j < 4; j++) {
                if (m == undefined) {
                    this.data.push(l[j * 3 + 0], l[j * 3 + 1], l[j * 3 + 2]);
                } else {
                    this.data.push(m[j * 3 + 0], m[j * 3 + 1], m[j * 3 + 2]);
                }
                if (n) {
                    this.data.push(o[j * 3 + 0], o[j * 3 + 1], o[j * 3 + 2]);
                }
                if (q == undefined) {
                    var h = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                    if (k) {
                        this.data.push(h[j * 4 + 0], h[j * 4 + 1], h[j * 4 + 2], h[j * 4 + 3]);
                    }
                } else {
                    if (k) {
                        this.data.push(q[j * 4 + 0], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3]);
                    }
                }
                if (p) {
                    this.data.push(g[j * 2 + 0], g[j * 2 + 1]);
                }
            }
        }
        return f;
    })();
    c.BoardModel = e;
    var a = (function () {
        function f(h, m, g, l, j, i, k) {
            if (k === void 0) {
                k = false;
            }
            this.verCrossSectionSmooth = h;
            this.horCrossSectionSmooth = m;
            this.verRadius = g;
            this.horRadius = l;
            this.data = new Array();
            this.index = new Array();
            this.normal = new Array();
            this.preCalculate(j, i, k);
        }
        f.prototype.preCalculate = function (l, s, k) {
            if (k === void 0) {
                k = false;
            }
            for (var o = 0; o <= this.verCrossSectionSmooth; o++) {
                var v = ((Math.PI * 2) / this.verCrossSectionSmooth) * o;
                var u = Math.cos(v);
                var t = Math.sin(v);
                for (var x = 0; x <= this.horCrossSectionSmooth; x++) {
                    var w = ((Math.PI * 2) / this.horCrossSectionSmooth) * x;
                    var r = (u * this.verRadius + this.horRadius) * Math.cos(w);
                    var p = t * this.verRadius;
                    var n = (u * this.verRadius + this.horRadius) * Math.sin(w);
                    this.data.push(r, p, n);
                    if (s) {
                        var q = u * Math.cos(w);
                        var m = u * Math.sin(w);
                        this.normal.push(q, t, m);
                        this.data.push(q, t, m);
                    }
                    if (l == undefined) {
                        var j = c.HSV2RGB((360 / this.horCrossSectionSmooth) * x, 1, 1, 1);
                        this.data.push(j[0], j[1], j[2], j[3]);
                    } else {
                        this.data.push(l[0], l[1], l[2], l[3]);
                    }
                    if (k) {
                        var h = (1 / this.horCrossSectionSmooth) * x;
                        var g = (1 / this.verCrossSectionSmooth) * o + 0.5;
                        if (g > 1) {
                            g -= 1;
                        }
                        g = 1 - g;
                        this.data.push(h, g);
                    }
                }
            }
            for (o = 0; o < this.verCrossSectionSmooth; o++) {
                for (x = 0; x < this.horCrossSectionSmooth; x++) {
                    v = (this.horCrossSectionSmooth + 1) * o + x;
                    this.index.push(v, v + this.horCrossSectionSmooth + 1, v + 1);
                    this.index.push(v + this.horCrossSectionSmooth + 1, v + this.horCrossSectionSmooth + 2, v + 1);
                }
            }
        };
        return f;
    })();
    c.TorusModel = a;
    var b = (function () {
        function f(h, l, g, j, i, k) {
            if (k === void 0) {
                k = false;
            }
            this.verCrossSectionSmooth = h;
            this.horCrossSectionSmooth = l;
            this.Radius = g;
            this.data = new Array();
            this.index = new Array();
            this.preCalculate(j, i, k);
        }
        f.prototype.preCalculate = function (j, q, h) {
            if (h === void 0) {
                h = false;
            }
            for (var m = 0; m <= this.verCrossSectionSmooth; m++) {
                var t = (Math.PI / this.verCrossSectionSmooth) * m;
                var s = Math.cos(t);
                var r = Math.sin(t);
                for (var v = 0; v <= this.horCrossSectionSmooth; v++) {
                    var u = ((Math.PI * 2) / this.horCrossSectionSmooth) * v;
                    var p = r * this.Radius * Math.cos(u);
                    var n = s * this.Radius;
                    var l = r * this.Radius * Math.sin(u);
                    this.data.push(p, n, l);
                    if (q) {
                        var o = r * Math.cos(u);
                        var k = r * Math.sin(u);
                        this.data.push(o, s, k);
                    }
                    if (j == undefined) {
                        var g = c.HSV2RGB((360 / this.horCrossSectionSmooth) * m, 1, 1, 1);
                        this.data.push(g[0], g[1], g[2], g[3]);
                    } else {
                        this.data.push(j[0], j[1], j[2], j[3]);
                    }
                    if (h) {
                        this.data.push(1 - (1 / this.horCrossSectionSmooth) * v, (1 / this.verCrossSectionSmooth) * m);
                    }
                }
            }
            for (m = 0; m < this.verCrossSectionSmooth; m++) {
                for (v = 0; v < this.horCrossSectionSmooth; v++) {
                    t = (this.horCrossSectionSmooth + 1) * m + v;
                    this.index.push(t, t + 1, t + this.horCrossSectionSmooth + 2);
                    this.index.push(t, t + this.horCrossSectionSmooth + 2, t + this.horCrossSectionSmooth + 1);
                }
            }
        };
        return f;
    })();
    c.ShpereModel = b;
    var d = (function () {
        function f(p, l, n, k) {
            if (k === void 0) {
                k = false;
            }
            this.side = p;
            this.data = new Array();
            this.index = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23];
            var r = p * 0.5;
            var q = [
                -r,
                -r,
                r,
                r,
                -r,
                r,
                r,
                r,
                r,
                -r,
                r,
                r,
                -r,
                -r,
                -r,
                -r,
                r,
                -r,
                r,
                r,
                -r,
                r,
                -r,
                -r,
                -r,
                r,
                -r,
                -r,
                r,
                r,
                r,
                r,
                r,
                r,
                r,
                -r,
                -r,
                -r,
                -r,
                r,
                -r,
                -r,
                r,
                -r,
                r,
                -r,
                -r,
                r,
                r,
                -r,
                -r,
                r,
                r,
                -r,
                r,
                r,
                r,
                r,
                -r,
                r,
                -r,
                -r,
                -r,
                -r,
                -r,
                r,
                -r,
                r,
                r,
                -r,
                r,
                -r,
            ];
            var o = [
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
            ];
            var h = new Array();
            for (var m = 0; m < q.length / 3; m++) {
                if (l != undefined) {
                    var j = l;
                } else {
                    j = c.HSV2RGB((360 / q.length / 3) * m, 1, 1, 1);
                }
                h.push(j[0], j[1], j[2], j[3]);
            }
            var s = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1];
            var g = 24;
            for (var m = 0; m < g; m++) {
                this.data.push(q[m * 3 + 0], q[m * 3 + 1], q[m * 3 + 2]);
                if (n) {
                    this.data.push(o[m * 3 + 0], o[m * 3 + 1], o[m * 3 + 2]);
                }
                this.data.push(h[m * 4 + 0], h[m * 4 + 1], h[m * 4 + 2], h[m * 4 + 3]);
                if (k) {
                    this.data.push(s[m * 2 + 0], s[m * 2 + 1]);
                }
            }
        }
        return f;
    })();
    c.CubeModel = d;
})(EcognitaMathLib || (EcognitaMathLib = {}));
var Utils;
(function (a) {
    var b = (function () {
        function c() {
            this.items = {};
        }
        c.prototype.set = function (d, e) {
            this.items[d] = e;
        };
        c.prototype["delete"] = function (d) {
            return delete this.items[d];
        };
        c.prototype.has = function (d) {
            return d in this.items;
        };
        c.prototype.get = function (d) {
            return this.items[d];
        };
        c.prototype.len = function () {
            return Object.keys(this.items).length;
        };
        c.prototype.forEach = function (e) {
            for (var d in this.items) {
                e(d, this.items[d]);
            }
        };
        return c;
    })();
    a.HashSet = b;
})(Utils || (Utils = {}));
var Utils;
(function (a) {
    var b = (function () {
        function c(d) {
            var e = this;
            this.gui = new dat.gui.GUI();
            this.data = d;
            this.gui.remember(d);
            this.uiController = new a.HashSet();
            this.folderHashSet = new a.HashSet();
            this.folderHashSet.set("f", "Filter");
            this.folderName = [];
            this.folderHashSet.forEach(function (g, f) {
                e.folderName.push(g);
            });
            this.initData();
            this.initFolder();
        }
        c.prototype.initFolder = function () {
            var d = this;
            this.folderName.forEach(function (h) {
                var i = d.gui.addFolder(d.folderHashSet.get(h));
                for (var g in d.data) {
                    var e = g.split("_");
                    if (g.includes("_") && e[0] == h) {
                        var j = i.add(d.data, g).listen();
                        d.uiController.set(g, j);
                    }
                }
            });
        };
        c.prototype.initData = function () {
            for (var d in this.data) {
                if (!d.includes("_")) {
                    this.gui.add(this.data, d);
                }
            }
        };
        return c;
    })();
    a.FilterViewerUI = b;
})(Utils || (Utils = {}));
var EcognitaWeb3D;
(function (b) {
    var a = (function () {
        function c(d) {
            this.chkWebGLEnv(d);
        }
        c.prototype.loadTexture = function (k, e, f, h, d, j) {
            var i = this;
            if (e === void 0) {
                e = false;
            }
            if (f === void 0) {
                f = gl.CLAMP_TO_EDGE;
            }
            if (h === void 0) {
                h = gl.LINEAR;
            }
            if (d === void 0) {
                d = true;
            }
            if (j === void 0) {
                j = 4;
            }
            var l = null;
            var g = EcognitaMathLib.imread(k);
            g.onload = function () {
                l = new EcognitaMathLib.WebGL_Texture(j, e, g, f, h, d);
                i.Texture.set(k, l);
            };
        };
        c.prototype.chkWebGLEnv = function (g) {
            this.canvas = g;
            try {
                gl = this.canvas.getContext("webgl") || this.canvas.getContext("experimental-webgl");
                this.stats = new Stats();
                document.body.appendChild(this.stats.dom);
            } catch (f) {}
            if (!gl) {
                throw new Error("Could not initialise WebGL");
            }
            var d = gl.getExtension("OES_texture_float");
            if (d == null) {
                throw new Error("float texture not supported");
            }
        };
        c.prototype.initGlobalVariables = function () {
            this.vbo = new Array();
            this.ibo = new Array();
            this.Texture = new Utils.HashSet();
            this.matUtil = new EcognitaMathLib.WebGLMatrix();
            this.quatUtil = new EcognitaMathLib.WebGLQuaternion();
        };
        c.prototype.initGlobalMatrix = function () {
            this.MATRIX = new Utils.HashSet();
            var d = this.matUtil;
            this.MATRIX.set("mMatrix", d.identity(d.create()));
            this.MATRIX.set("vMatrix", d.identity(d.create()));
            this.MATRIX.set("pMatrix", d.identity(d.create()));
            this.MATRIX.set("vpMatrix", d.identity(d.create()));
            this.MATRIX.set("mvpMatrix", d.identity(d.create()));
            this.MATRIX.set("invMatrix", d.identity(d.create()));
        };
        c.prototype.loadExtraLibrary = function (d) {
            this.ui_data = d;
            this.uiUtil = new Utils.FilterViewerUI(this.ui_data);
            this.extHammer = new EcognitaMathLib.Hammer_Utils(this.canvas);
        };
        c.prototype.loadInternalLibrary = function (d) {
            var e = this;
            this.framebuffers = new Utils.HashSet();
            this.shaders = new Utils.HashSet();
            this.uniLocations = new Utils.HashSet();
            d.forEach(function (f) {
                var g = new EcognitaMathLib.WebGL_Shader(Shaders, f.name + "-vert", f.name + "-frag");
                e.shaders.set(f.name, g);
                e.uniLocations.set(f.name, new Array());
            });
        };
        c.prototype.settingFrameBuffer = function (d) {
            var g = this.canvas.width;
            var f = this.canvas.height;
            var e = new EcognitaMathLib.WebGL_FrameBuffer(g, f);
            e.bindFrameBuffer();
            e.bindDepthBuffer();
            e.renderToFloatTexure();
            e.release();
            this.framebuffers.set(d, e);
        };
        c.prototype.renderSceneByFrameBuffer = function (f, e, d) {
            if (d === void 0) {
                d = gl.TEXTURE0;
            }
            f.bindFrameBuffer();
            e();
            gl.activeTexture(d);
            gl.bindTexture(gl.TEXTURE_2D, f.targetTexture);
        };
        c.prototype.renderBoardByFrameBuffer = function (h, j, e, g, i, d, f) {
            if (i === void 0) {
                i = false;
            }
            if (d === void 0) {
                d = gl.TEXTURE0;
            }
            if (f === void 0) {
                f = undefined;
            }
            h.bind();
            if (i) {
                f.bindFrameBuffer();
            } else {
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            }
            gl.clearColor(0, 0, 0, 1);
            gl.clearDepth(1);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            j.bind(h);
            e.bind();
            g();
            e.draw(gl.TRIANGLES);
            if (i) {
                gl.activeTexture(d);
                gl.bindTexture(gl.TEXTURE_2D, f.targetTexture);
            }
        };
        return c;
    })();
    b.WebGLEnv = a;
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
var EcognitaWeb3D;
(function (b) {
    var a = (function (d) {
        __extends(c, d);
        function c(e) {
            return d.call(this, e) || this;
        }
        c.prototype.loadAssets = function () {
            this.loadTexture("./image/k0.png", true, gl.CLAMP_TO_BORDER, gl.NEAREST, false);
            this.loadTexture("./image/visual_rgb.png");
            this.loadTexture("./image/lion.png", false);
            this.loadTexture("./image/anim.png", false);
            this.loadTexture("./image/cat.jpg", false);
            this.loadTexture("./image/woman.png", false, gl.CLAMP_TO_EDGE, gl.LINEAR, false);
            this.loadTexture("./image/noise.png", false);
        };
        c.prototype.getReqQuery = function () {
            if (window.location.href.split("?").length == 1) {
                return {};
            }
            var k = window.location.href.split("?")[1];
            var j = {};
            if (k != "") {
                var e = k.split("&");
                for (var g = 0; g < e.length; g++) {
                    var f = e[g].split("=")[0];
                    var h = e[g].split("=")[1];
                    j[f] = h;
                }
            }
            return j;
        };
        c.prototype.regisButton = function (e) {
            var f = this;
            this.btnStatusList = new Utils.HashSet();
            e.forEach(function (g) {
                f.btnStatusList.set(g.name, f.ui_data[g.name]);
                f.uiUtil.uiController.get(g.name).onChange(function (h) {
                    f.usrSelectChange(g.name, h, b.RenderPipeLine[g.pipline], b.Filter[g.filter], g.shader);
                });
            });
        };
        c.prototype.regisUniforms = function (e) {
            var f = this;
            e.forEach(function (g) {
                var h = new Array();
                var i = g.uniforms;
                i.forEach(function (j) {
                    h.push(j);
                });
                f.settingUniform(g.name, h);
            });
        };
        c.prototype.regisUserParam = function (f) {
            this.filterMvpMatrix = this.matUtil.identity(this.matUtil.create());
            this.usrParams = f.user_params;
            this.usrSelected = f.user_selected;
            var e = f.default_btn;
            var g = this.getReqQuery();
            if (g.p == null || g.f == null || g.s == null || g.b == null) {
                this.usrPipeLine = b.RenderPipeLine[f.default_pipline];
                this.usrFilter = b.Filter[f.default_filter];
                this.filterShader = this.shaders.get(f.default_shader);
            } else {
                this.usrPipeLine = b.RenderPipeLine[g.p];
                this.usrFilter = b.Filter[g.f];
                this.filterShader = this.shaders.get(g.s);
                e = g.b;
            }
            this.uiData[e] = true;
        };
        c.prototype.initialize = function (e, h, g, f) {
            this.uiData = e;
            this.initGlobalVariables();
            this.loadAssets();
            this.loadInternalLibrary(h.shaderList);
            this.regisUniforms(h.shaderList);
            this.regisUserParam(f);
            this.loadExtraLibrary(e);
            this.initGlobalMatrix();
            this.regisButton(g.buttonList);
            this.initModel();
            this.regisEvent();
            this.settingRenderPipeline();
            this.regisAnimeFunc();
        };
        c.prototype.initModel = function () {
            var f = new EcognitaMathLib.TorusModel(64, 64, 1, 2, [1, 1, 1, 1], true, false);
            var i = new EcognitaMathLib.WebGL_VertexBuffer();
            var g = new EcognitaMathLib.WebGL_IndexBuffer();
            this.vbo.push(i);
            this.ibo.push(g);
            i.addAttribute("position", 3, gl.FLOAT, false);
            i.addAttribute("normal", 3, gl.FLOAT, false);
            i.addAttribute("color", 4, gl.FLOAT, false);
            i.init(f.data.length / 10);
            i.copy(f.data);
            g.init(f.index);
            var e = [-1, 1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0];
            var k = new EcognitaMathLib.BoardModel(e, undefined, false, false, true);
            var j = new EcognitaMathLib.WebGL_VertexBuffer();
            var h = new EcognitaMathLib.WebGL_IndexBuffer();
            this.vbo.push(j);
            this.ibo.push(h);
            j.addAttribute("position", 3, gl.FLOAT, false);
            j.addAttribute("texCoord", 2, gl.FLOAT, false);
            k.index = [0, 2, 1, 2, 3, 1];
            j.init(k.data.length / 5);
            j.copy(k.data);
            h.init(k.index);
        };
        c.prototype.renderGaussianFilter = function (g, f, h) {
            if (h === void 0) {
                h = 0;
            }
            var e = this.uniLocations.get("gaussianFilter");
            gl.uniformMatrix4fv(e[0], false, this.filterMvpMatrix);
            gl.uniform1i(e[1], h);
            gl.uniform1fv(e[2], this.usrParams.gaussianWeight);
            gl.uniform1i(e[3], g);
            gl.uniform1f(e[4], this.canvas.height);
            gl.uniform1f(e[5], this.canvas.width);
            gl.uniform1i(e[6], f);
        };
        c.prototype.renderFilter = function () {
            if (this.usrFilter == b.Filter.LAPLACIAN) {
                var n = this.uniLocations.get("laplacianFilter");
                gl.uniformMatrix4fv(n[0], false, this.filterMvpMatrix);
                gl.uniform1i(n[1], 0);
                gl.uniform1fv(n[2], this.usrParams.laplacianCoef);
                gl.uniform1f(n[3], this.canvas.height);
                gl.uniform1f(n[4], this.canvas.width);
                gl.uniform1i(n[5], this.btnStatusList.get("f_LaplacianFilter"));
            } else {
                if (this.usrFilter == b.Filter.SOBEL) {
                    var i = this.uniLocations.get("sobelFilter");
                    gl.uniformMatrix4fv(i[0], false, this.filterMvpMatrix);
                    gl.uniform1i(i[1], 0);
                    gl.uniform1fv(i[2], this.usrParams.sobelHorCoef);
                    gl.uniform1fv(i[3], this.usrParams.sobelVerCoef);
                    gl.uniform1f(i[4], this.canvas.height);
                    gl.uniform1f(i[5], this.canvas.width);
                    gl.uniform1i(i[6], this.btnStatusList.get("f_SobelFilter"));
                } else {
                    if (this.usrFilter == b.Filter.DoG) {
                        var m = this.uniLocations.get("DoG");
                        gl.uniformMatrix4fv(m[0], false, this.filterMvpMatrix);
                        gl.uniform1i(m[1], 0);
                        gl.uniform1f(m[2], 1);
                        gl.uniform1f(m[3], 1.6);
                        gl.uniform1f(m[4], 0.99);
                        gl.uniform1f(m[5], 2);
                        gl.uniform1f(m[6], this.canvas.height);
                        gl.uniform1f(m[7], this.canvas.width);
                        gl.uniform1i(m[8], this.btnStatusList.get("f_DoG"));
                    } else {
                        if (this.usrFilter == b.Filter.XDoG) {
                            var f = this.uniLocations.get("XDoG");
                            gl.uniformMatrix4fv(f[0], false, this.filterMvpMatrix);
                            gl.uniform1i(f[1], 0);
                            gl.uniform1f(f[2], 1.4);
                            gl.uniform1f(f[3], 1.6);
                            gl.uniform1f(f[4], 21.7);
                            gl.uniform1f(f[5], 79.5);
                            gl.uniform1f(f[6], 0.017);
                            gl.uniform1f(f[7], this.canvas.height);
                            gl.uniform1f(f[8], this.canvas.width);
                            gl.uniform1i(f[9], this.btnStatusList.get("f_XDoG"));
                        } else {
                            if (this.usrFilter == b.Filter.KUWAHARA) {
                                var p = this.uniLocations.get("kuwaharaFilter");
                                gl.uniformMatrix4fv(p[0], false, this.filterMvpMatrix);
                                gl.uniform1i(p[1], 0);
                                gl.uniform1f(p[2], this.canvas.height);
                                gl.uniform1f(p[3], this.canvas.width);
                                gl.uniform1i(p[4], this.btnStatusList.get("f_KuwaharaFilter"));
                            } else {
                                if (this.usrFilter == b.Filter.GKUWAHARA) {
                                    var h = this.uniLocations.get("gkuwaharaFilter");
                                    gl.uniformMatrix4fv(h[0], false, this.filterMvpMatrix);
                                    gl.uniform1i(h[1], 0);
                                    gl.uniform1fv(h[2], this.usrParams.gkweight);
                                    gl.uniform1f(h[3], this.canvas.height);
                                    gl.uniform1f(h[4], this.canvas.width);
                                    gl.uniform1i(h[5], this.btnStatusList.get("f_GeneralizedKuwaharaFilter"));
                                } else {
                                    if (this.usrFilter == b.Filter.ANISTROPIC) {
                                        var k = this.uniLocations.get("Anisotropic");
                                        gl.uniformMatrix4fv(k[0], false, this.filterMvpMatrix);
                                        gl.uniform1i(k[1], 0);
                                        gl.uniform1i(k[2], 1);
                                        gl.uniform1i(k[3], 2);
                                        gl.uniform1f(k[4], this.canvas.height);
                                        gl.uniform1f(k[5], this.canvas.width);
                                        gl.uniform1i(k[6], this.btnStatusList.get("f_VisualAnisotropic"));
                                    } else {
                                        if (this.usrFilter == b.Filter.AKUWAHARA) {
                                            var g = this.uniLocations.get("AKF");
                                            gl.uniformMatrix4fv(g[0], false, this.filterMvpMatrix);
                                            gl.uniform1i(g[1], 0);
                                            gl.uniform1i(g[2], 1);
                                            gl.uniform1i(g[3], 2);
                                            gl.uniform1f(g[4], 6);
                                            gl.uniform1f(g[5], 8);
                                            gl.uniform1f(g[6], 1);
                                            gl.uniform1f(g[7], this.canvas.height);
                                            gl.uniform1f(g[8], this.canvas.width);
                                            gl.uniform1i(g[9], this.btnStatusList.get("f_AnisotropicKuwahara"));
                                        } else {
                                            if (this.usrFilter == b.Filter.LIC || this.usrFilter == b.Filter.NOISELIC) {
                                                var e = this.uniLocations.get("LIC");
                                                gl.uniformMatrix4fv(e[0], false, this.filterMvpMatrix);
                                                gl.uniform1i(e[1], 0);
                                                gl.uniform1i(e[2], 1);
                                                gl.uniform1f(e[3], 3);
                                                gl.uniform1f(e[4], this.canvas.height);
                                                gl.uniform1f(e[5], this.canvas.width);
                                                if (this.usrFilter == b.Filter.LIC) {
                                                    gl.uniform1i(e[6], this.btnStatusList.get("f_LIC"));
                                                } else {
                                                    if (this.usrFilter == b.Filter.NOISELIC) {
                                                        gl.uniform1i(e[6], this.btnStatusList.get("f_NoiseLIC"));
                                                    }
                                                }
                                            } else {
                                                if (this.usrFilter == b.Filter.FDoG) {
                                                    var l = this.uniLocations.get("FDoG");
                                                    gl.uniformMatrix4fv(l[0], false, this.filterMvpMatrix);
                                                    gl.uniform1i(l[1], 0);
                                                    gl.uniform1i(l[2], 1);
                                                    gl.uniform1f(l[3], 3);
                                                    gl.uniform1f(l[4], 2);
                                                    gl.uniform1f(l[5], this.canvas.height);
                                                    gl.uniform1f(l[6], this.canvas.width);
                                                    gl.uniform1i(l[7], this.btnStatusList.get("f_FDoG"));
                                                } else {
                                                    if (this.usrFilter == b.Filter.FXDoG) {
                                                        var j = this.uniLocations.get("FXDoG");
                                                        gl.uniformMatrix4fv(j[0], false, this.filterMvpMatrix);
                                                        gl.uniform1i(j[1], 0);
                                                        gl.uniform1i(j[2], 1);
                                                        gl.uniform1f(j[3], 4.4);
                                                        gl.uniform1f(j[4], 0.017);
                                                        gl.uniform1f(j[5], 79.5);
                                                        gl.uniform1f(j[6], this.canvas.height);
                                                        gl.uniform1f(j[7], this.canvas.width);
                                                        gl.uniform1i(j[8], this.btnStatusList.get("f_FXDoG"));
                                                    } else {
                                                        if (this.usrFilter == b.Filter.ABSTRACTION) {
                                                            var o = this.uniLocations.get("Abstraction");
                                                            gl.uniformMatrix4fv(o[0], false, this.filterMvpMatrix);
                                                            gl.uniform1i(o[1], 1);
                                                            gl.uniform1i(o[2], 3);
                                                            gl.uniform1i(o[3], 4);
                                                            gl.uniform3fv(o[4], [0, 0, 0]);
                                                            gl.uniform1f(o[5], this.canvas.height);
                                                            gl.uniform1f(o[6], this.canvas.width);
                                                            gl.uniform1i(o[7], this.btnStatusList.get("f_Abstraction"));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        c.prototype.settingUniform = function (h, g) {
            var f = this.uniLocations.get(h);
            var e = this.shaders.get(h);
            g.forEach(function (i) {
                f.push(e.uniformIndex(i));
            });
        };
        c.prototype.settingRenderPipeline = function () {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
            gl.enable(gl.CULL_FACE);
        };
        c.prototype.usrSelectChange = function (k, j, f, h, i) {
            this.btnStatusList.set(k, j);
            if (j) {
                this.usrPipeLine = f;
                this.usrFilter = h;
                this.filterShader = this.shaders.get(i);
                for (var g in this.ui_data) {
                    if (g.includes("_")) {
                        var e = g.split("_");
                        if (e[0] == "f" && g != k) {
                            this.btnStatusList.set(g, !j);
                            this.ui_data[g] = !j;
                        }
                    }
                }
            }
        };
        c.prototype.regisEvent = function () {
            var i = this;
            $('[data-toggle="tooltip"]').tooltip();
            $("select").imagepicker({
                hide_select: true,
                show_label: false,
                selected: function () {
                    i.usrSelected = $("select").val();
                },
            });
            this.canvas.addEventListener("dragover", function (j) {
                j.preventDefault();
            });
            this.canvas.addEventListener("drop", function (l) {
                l.preventDefault();
                var k = l.dataTransfer.files[0];
                if (!k.type.match("image.*")) {
                    alert("you should upload a image!");
                    return;
                }
                var j = new FileReader();
                j.onload = function () {
                    i.loadTexture(j.result, false, gl.CLAMP_TO_EDGE, gl.LINEAR, false);
                    i.usrSelected = j.result;
                };
                j.readAsDataURL(k);
            });
            var h = 0;
            var g = 0;
            var f = false;
            var e = this.extHammer;
            e.on_pan = function (t) {
                var p = t.target;
                if (!f) {
                    f = true;
                    h = p.offsetLeft;
                    g = p.offsetTop;
                }
                var n = t.center.x - h;
                var k = t.center.y - g;
                var q = i.canvas.width;
                var l = i.canvas.height;
                var o = 1 / Math.sqrt(q * q + l * l);
                var u = n - q * 0.5;
                var s = k - l * 0.5;
                var m = Math.sqrt(u * u + s * s);
                var j = m * 2 * Math.PI * o;
                if (m != 1) {
                    m = 1 / m;
                    u *= m;
                    s *= m;
                }
                i.usrQuaternion = i.quatUtil.rotate(j, [s, u, 0]);
                if (t.isFinal) {
                    f = false;
                }
            };
            e.enablePan();
        };
        c.prototype.regisFrameBuffer = function (e) {
            var g = Array(e);
            for (var f = 0; f < e; f++) {
                this.settingFrameBuffer("frameBuffer" + f);
                var h = this.framebuffers.get("frameBuffer" + f);
                g[f] = h;
            }
            return g;
        };
        c.prototype.regisAnimeFunc = function () {
            var l = this;
            var z = 0;
            var E = 0;
            var V = [-0.577, 0.577, 0.577];
            var P = this.matUtil;
            var N = this.quatUtil;
            this.usrQuaternion = N.identity(N.create());
            var n = this.shaders.get("filterScene");
            var B = this.uniLocations.get("filterScene");
            var A = this.vbo[0];
            var U = this.ibo[0];
            var C = this.vbo[1];
            var X = this.ibo[1];
            var w = this.MATRIX.get("mMatrix");
            var x = this.MATRIX.get("vMatrix");
            var v = this.MATRIX.get("pMatrix");
            var G = this.MATRIX.get("vpMatrix");
            var i = this.MATRIX.get("mvpMatrix");
            var L = this.MATRIX.get("invMatrix");
            var J = this.shaders.get("specCpt");
            var K = this.uniLocations.get("specCpt");
            var f = this.shaders.get("synth");
            var y = this.uniLocations.get("synth");
            var Q = this.shaders.get("luminance");
            var h = this.uniLocations.get("luminance");
            var s = this.shaders.get("TF");
            var o = this.uniLocations.get("TF");
            var F = this.shaders.get("ETF");
            var j = this.uniLocations.get("ETF");
            var M = this.shaders.get("P_FDoG");
            var S = this.uniLocations.get("P_FDoG");
            var t = this.shaders.get("P_FXDoG");
            var W = this.uniLocations.get("P_FXDoG");
            var p = this.shaders.get("FXDoG");
            var g = this.uniLocations.get("FXDoG");
            var D = this.shaders.get("SST");
            var H = this.uniLocations.get("SST");
            var e = this.shaders.get("Gaussian_K");
            var R = this.uniLocations.get("Gaussian_K");
            var T = this.shaders.get("TFM");
            var O = this.uniLocations.get("TFM");
            var r = this.shaders.get("AKF");
            var u = this.uniLocations.get("AKF");
            var I = this.regisFrameBuffer(5);
            var k = function () {
                l.stats.begin();
                z++;
                if (z % 2 == 0) {
                    E++;
                }
                var ad = ((z % 360) * Math.PI) / 180;
                var m = new Array();
                var ac = new Array();
                m = N.ToV3([0, 20, 0], l.usrQuaternion);
                ac = N.ToV3([0, 0, -1], l.usrQuaternion);
                x = P.viewMatrix(m, [0, 0, 0], ac);
                v = P.perspectiveMatrix(90, l.canvas.width / l.canvas.height, 0.1, 100);
                G = P.multiply(v, x);
                x = P.viewMatrix([0, 0, 0.5], [0, 0, 0], [0, 1, 0]);
                v = P.orthoMatrix(-1, 1, 1, -1, 0.1, 1);
                l.filterMvpMatrix = P.multiply(v, x);
                var q = l.Texture.get(l.usrSelected);
                if (l.usrPipeLine == b.RenderPipeLine.CONVOLUTION_FILTER) {
                    if (q != undefined && l.ui_data.useTexture) {
                        gl.activeTexture(gl.TEXTURE0);
                        q.bind(q.texture);
                    } else {
                        l.renderSceneByFrameBuffer(I[0], Z);
                    }
                    l.renderBoardByFrameBuffer(l.filterShader, C, X, function () {
                        l.renderFilter();
                    });
                } else {
                    if (l.usrPipeLine == b.RenderPipeLine.BLOOM_EFFECT) {
                        if (q != undefined && l.ui_data.useTexture) {
                            gl.activeTexture(gl.TEXTURE0);
                            q.bind(q.texture);
                        } else {
                            l.renderSceneByFrameBuffer(I[0], Z);
                        }
                        l.renderBoardByFrameBuffer(
                            Q,
                            C,
                            X,
                            function () {
                                gl.uniformMatrix4fv(h[0], false, l.filterMvpMatrix);
                                gl.uniform1i(h[1], 0);
                                gl.uniform1f(h[2], 0.5);
                            },
                            true,
                            gl.TEXTURE1,
                            I[1]
                        );
                        var af = 9;
                        for (var ab = 0; ab < af; ab++) {
                            l.renderBoardByFrameBuffer(
                                l.filterShader,
                                C,
                                X,
                                function () {
                                    l.renderGaussianFilter(true, l.btnStatusList.get("f_BloomEffect"), 1);
                                },
                                true,
                                gl.TEXTURE0,
                                I[0]
                            );
                            l.renderBoardByFrameBuffer(
                                l.filterShader,
                                C,
                                X,
                                function () {
                                    l.renderGaussianFilter(false, l.btnStatusList.get("f_BloomEffect"));
                                },
                                true,
                                gl.TEXTURE1,
                                I[1]
                            );
                        }
                        if (q != undefined && l.ui_data.useTexture) {
                            gl.activeTexture(gl.TEXTURE0);
                            q.bind(q.texture);
                        } else {
                            l.renderBoardByFrameBuffer(
                                l.filterShader,
                                C,
                                X,
                                function () {
                                    Z();
                                },
                                true,
                                gl.TEXTURE0,
                                I[0]
                            );
                        }
                        l.renderBoardByFrameBuffer(f, C, X, function () {
                            gl.uniformMatrix4fv(y[0], false, l.filterMvpMatrix);
                            gl.uniform1i(y[1], 0);
                            gl.uniform1i(y[2], 1);
                            gl.uniform1i(y[3], l.btnStatusList.get("f_BloomEffect"));
                        });
                    } else {
                        if (l.usrPipeLine == b.RenderPipeLine.CONVOLUTION_TWICE) {
                            if (q != undefined && l.ui_data.useTexture) {
                                gl.activeTexture(gl.TEXTURE0);
                                q.bind(q.texture);
                            } else {
                                l.renderSceneByFrameBuffer(I[0], Z);
                            }
                            if (l.btnStatusList.get("f_GaussianFilter")) {
                                l.renderBoardByFrameBuffer(
                                    l.filterShader,
                                    C,
                                    X,
                                    function () {
                                        l.renderGaussianFilter(true, l.btnStatusList.get("f_GaussianFilter"));
                                    },
                                    true,
                                    gl.TEXTURE0,
                                    I[1]
                                );
                                l.renderBoardByFrameBuffer(l.filterShader, C, X, function () {
                                    l.renderGaussianFilter(false, l.btnStatusList.get("f_GaussianFilter"));
                                });
                            } else {
                                l.renderBoardByFrameBuffer(l.filterShader, C, X, function () {
                                    l.renderGaussianFilter(false, l.btnStatusList.get("f_GaussianFilter"));
                                });
                            }
                        } else {
                            if (l.usrPipeLine == b.RenderPipeLine.ANISTROPIC) {
                                if (l.usrFilter == b.Filter.ANISTROPIC) {
                                    var ag = l.Texture.get("./image/visual_rgb.png");
                                    if (ag != undefined) {
                                        gl.activeTexture(gl.TEXTURE2);
                                        ag.bind(ag.texture);
                                    }
                                } else {
                                    if (l.usrFilter == b.Filter.AKUWAHARA) {
                                        var Y = l.Texture.get("./image/k0.png");
                                        if (Y != undefined) {
                                            gl.activeTexture(gl.TEXTURE2);
                                            Y.bind(Y.texture);
                                        }
                                    }
                                }
                                if (q != undefined && l.ui_data.useTexture) {
                                    gl.activeTexture(gl.TEXTURE0);
                                    q.bind(q.texture);
                                } else {
                                    l.renderSceneByFrameBuffer(I[0], Z);
                                }
                                l.renderBoardByFrameBuffer(
                                    D,
                                    C,
                                    X,
                                    function () {
                                        gl.uniformMatrix4fv(H[0], false, l.filterMvpMatrix);
                                        gl.uniform1i(H[1], 0);
                                        gl.uniform1f(H[2], l.canvas.height);
                                        gl.uniform1f(H[3], l.canvas.width);
                                    },
                                    true,
                                    gl.TEXTURE0,
                                    I[1]
                                );
                                l.renderBoardByFrameBuffer(
                                    e,
                                    C,
                                    X,
                                    function () {
                                        gl.uniformMatrix4fv(R[0], false, l.filterMvpMatrix);
                                        gl.uniform1i(R[1], 0);
                                        gl.uniform1f(R[2], 2);
                                        gl.uniform1f(R[3], l.canvas.height);
                                        gl.uniform1f(R[4], l.canvas.width);
                                    },
                                    true,
                                    gl.TEXTURE0,
                                    I[0]
                                );
                                l.renderBoardByFrameBuffer(
                                    T,
                                    C,
                                    X,
                                    function () {
                                        gl.uniformMatrix4fv(O[0], false, l.filterMvpMatrix);
                                        gl.uniform1i(O[1], 0);
                                        gl.uniform1f(O[2], l.canvas.height);
                                        gl.uniform1f(O[3], l.canvas.width);
                                    },
                                    true,
                                    gl.TEXTURE0,
                                    I[1]
                                );
                                if (l.usrFilter == b.Filter.NOISELIC) {
                                    gl.activeTexture(gl.TEXTURE1);
                                    var aa = l.Texture.get("./image/noise.png");
                                    if (aa != undefined) {
                                        aa.bind(aa.texture);
                                    }
                                } else {
                                    if (q != undefined && l.ui_data.useTexture) {
                                        gl.activeTexture(gl.TEXTURE1);
                                        q.bind(q.texture);
                                    } else {
                                        l.renderSceneByFrameBuffer(I[0], Z, gl.TEXTURE1);
                                    }
                                }
                                if (l.usrFilter == b.Filter.FDoG) {
                                    l.renderBoardByFrameBuffer(
                                        M,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(S[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(S[1], 0);
                                            gl.uniform1i(S[2], 1);
                                            gl.uniform1f(S[3], 1);
                                            gl.uniform1f(S[4], 1.6);
                                            gl.uniform1f(S[5], 0.99);
                                            gl.uniform1f(S[6], l.canvas.height);
                                            gl.uniform1f(S[7], l.canvas.width);
                                            gl.uniform1i(S[8], l.btnStatusList.get("f_FDoG"));
                                        },
                                        true,
                                        gl.TEXTURE1,
                                        I[2]
                                    );
                                } else {
                                    if (l.usrFilter == b.Filter.FXDoG) {
                                        l.renderBoardByFrameBuffer(
                                            t,
                                            C,
                                            X,
                                            function () {
                                                gl.uniformMatrix4fv(W[0], false, l.filterMvpMatrix);
                                                gl.uniform1i(W[1], 0);
                                                gl.uniform1i(W[2], 1);
                                                gl.uniform1f(W[3], 1.4);
                                                gl.uniform1f(W[4], 1.6);
                                                gl.uniform1f(W[5], 21.7);
                                                gl.uniform1f(W[6], l.canvas.height);
                                                gl.uniform1f(W[7], l.canvas.width);
                                                gl.uniform1i(W[8], l.btnStatusList.get("f_FXDoG"));
                                            },
                                            true,
                                            gl.TEXTURE1,
                                            I[2]
                                        );
                                    }
                                }
                                l.renderBoardByFrameBuffer(l.filterShader, C, X, function () {
                                    l.renderFilter();
                                });
                            } else {
                                if (l.usrPipeLine == b.RenderPipeLine.ABSTRACTION) {
                                    var Y = l.Texture.get("./image/k0.png");
                                    if (Y != undefined) {
                                        gl.activeTexture(gl.TEXTURE2);
                                        Y.bind(Y.texture);
                                    }
                                    if (q != undefined && l.ui_data.useTexture) {
                                        gl.activeTexture(gl.TEXTURE0);
                                        q.bind(q.texture);
                                    } else {
                                        l.renderSceneByFrameBuffer(I[0], Z);
                                    }
                                    l.renderBoardByFrameBuffer(
                                        D,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(H[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(H[1], 0);
                                            gl.uniform1f(H[2], l.canvas.height);
                                            gl.uniform1f(H[3], l.canvas.width);
                                        },
                                        true,
                                        gl.TEXTURE0,
                                        I[1]
                                    );
                                    l.renderBoardByFrameBuffer(
                                        e,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(R[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(R[1], 0);
                                            gl.uniform1f(R[2], 2);
                                            gl.uniform1f(R[3], l.canvas.height);
                                            gl.uniform1f(R[4], l.canvas.width);
                                        },
                                        true,
                                        gl.TEXTURE0,
                                        I[0]
                                    );
                                    l.renderBoardByFrameBuffer(
                                        T,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(O[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(O[1], 0);
                                            gl.uniform1f(O[2], l.canvas.height);
                                            gl.uniform1f(O[3], l.canvas.width);
                                        },
                                        true,
                                        gl.TEXTURE0,
                                        I[1]
                                    );
                                    if (q != undefined && l.ui_data.useTexture) {
                                        gl.activeTexture(gl.TEXTURE1);
                                        q.bind(q.texture);
                                    } else {
                                        l.renderSceneByFrameBuffer(I[0], Z, gl.TEXTURE1);
                                    }
                                    l.renderBoardByFrameBuffer(
                                        r,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(u[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(u[1], 0);
                                            gl.uniform1i(u[2], 1);
                                            gl.uniform1i(u[3], 2);
                                            gl.uniform1f(u[4], 6);
                                            gl.uniform1f(u[5], 8);
                                            gl.uniform1f(u[6], 1);
                                            gl.uniform1f(u[7], l.canvas.height);
                                            gl.uniform1f(u[8], l.canvas.width);
                                            gl.uniform1i(u[9], l.btnStatusList.get("f_Abstraction"));
                                        },
                                        true,
                                        gl.TEXTURE3,
                                        I[2]
                                    );
                                    l.renderBoardByFrameBuffer(
                                        t,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(W[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(W[1], 0);
                                            gl.uniform1i(W[2], 1);
                                            gl.uniform1f(W[3], 1.4);
                                            gl.uniform1f(W[4], 1.6);
                                            gl.uniform1f(W[5], 21.7);
                                            gl.uniform1f(W[6], l.canvas.height);
                                            gl.uniform1f(W[7], l.canvas.width);
                                            gl.uniform1i(W[8], l.btnStatusList.get("f_Abstraction"));
                                        },
                                        true,
                                        gl.TEXTURE4,
                                        I[3]
                                    );
                                    l.renderBoardByFrameBuffer(
                                        p,
                                        C,
                                        X,
                                        function () {
                                            gl.uniformMatrix4fv(g[0], false, l.filterMvpMatrix);
                                            gl.uniform1i(g[1], 0);
                                            gl.uniform1i(g[2], 4);
                                            gl.uniform1f(g[3], 4.4);
                                            gl.uniform1f(g[4], 0.017);
                                            gl.uniform1f(g[5], 79.5);
                                            gl.uniform1f(g[6], l.canvas.height);
                                            gl.uniform1f(g[7], l.canvas.width);
                                            gl.uniform1i(g[8], l.btnStatusList.get("f_Abstraction"));
                                        },
                                        true,
                                        gl.TEXTURE4,
                                        I[4]
                                    );
                                    l.renderBoardByFrameBuffer(l.filterShader, C, X, function () {
                                        l.renderFilter();
                                    });
                                }
                            }
                        }
                    }
                }
                gl.flush();
                l.stats.end();
                requestAnimationFrame(k);
                function Z() {
                    n.bind();
                    var ah = EcognitaMathLib.HSV2RGB(E % 360, 1, 1, 1);
                    gl.clearColor(ah[0], ah[1], ah[2], ah[3]);
                    gl.clearDepth(1);
                    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    A.bind(n);
                    U.bind();
                    for (var ai = 0; ai < 9; ai++) {
                        var aj = EcognitaMathLib.HSV2RGB(ai * 40, 1, 1, 1);
                        w = P.identity(w);
                        w = P.rotate(w, (ai * 2 * Math.PI) / 9, [0, 1, 0]);
                        w = P.translate(w, [0, 0, 10]);
                        w = P.rotate(w, ad, [1, 1, 0]);
                        i = P.multiply(G, w);
                        L = P.inverse(w);
                        gl.uniformMatrix4fv(B[0], false, i);
                        gl.uniformMatrix4fv(B[1], false, L);
                        gl.uniform3fv(B[2], V);
                        gl.uniform3fv(B[3], m);
                        gl.uniform4fv(B[4], aj);
                        U.draw(gl.TRIANGLES);
                    }
                }
                function ae() {
                    J.bind();
                    gl.clearColor(0, 0, 0, 1);
                    gl.clearDepth(1);
                    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    A.bind(J);
                    U.bind();
                    for (var ah = 0; ah < 9; ah++) {
                        w = P.identity(w);
                        w = P.rotate(w, (ah * 2 * Math.PI) / 9, [0, 1, 0]);
                        w = P.translate(w, [0, 0, 10]);
                        w = P.rotate(w, ad, [1, 1, 0]);
                        i = P.multiply(G, w);
                        L = P.inverse(w);
                        gl.uniformMatrix4fv(K[0], false, i);
                        gl.uniformMatrix4fv(K[1], false, L);
                        gl.uniform3fv(K[2], V);
                        gl.uniform3fv(K[3], m);
                        U.draw(gl.TRIANGLES);
                    }
                }
            };
            k();
        };
        return c;
    })(b.WebGLEnv);
    b.FilterViewer = a;
})(EcognitaWeb3D || (EcognitaWeb3D = {}));
var viewer = document.getElementById("canvas_viewer");
viewer.width = 1920;
viewer.height = 1080;
$.getJSON("./config/ui.json", function (a) {
    $.getJSON("./config/shader.json", function (b) {
        $.getJSON("./config/button.json", function (c) {
            $.getJSON("./config/user_config.json", function (e) {
                var d = new EcognitaWeb3D.FilterViewer(viewer);
                d.initialize(a, b, c, e);
            });
        });
    });
});
