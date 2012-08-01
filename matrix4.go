/*
This code is an incomplete port of the C++ algebra library WildMagic5 (geometrictools.com)
Note that this code uses column major matrixes, just like OpenGl
Distributed under the Boost Software License, Version 1.0.
http://www.boost.org/LICENSE_1_0.txt
http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
*/

package math3d

import (
	"errors"
	"fmt"
)

// This is a 4x4 matrix of float32, stored in OpenGl format. Note - it's not rowmajor
type Matrix4 []float32

func NewMatrix4V(v []float32, rowMajor bool) (Matrix4, error) {
	if len(v) != 16 {
		return Matrix4{}, errors.New("expecting 16 elements in value list")
	}
	if rowMajor {
		// transform the data to OpenGl format
		return Matrix4{
			v[0], v[4], v[8], v[12],
			v[1], v[5], v[9], v[13],
			v[2], v[6], v[10], v[14],
			v[3], v[7], v[11], v[15],
		}, nil
	}
	m := NewMatrix4()
	copy(m, v)
	return m, nil
}

func NewMatrix4() Matrix4 {
	return Matrix4{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

/*
// FIXME untested code
func newRotationMatrix(look, tmpUp Vector3) Matrix4 {
	look = look.Copy().Normalize()
	right := tmpUp.Copy().Normalize().Cross(look).Normalize()
	up := look.Cross(right).Normalize()

	return Matrix4{
		right[0], right[1], right[2], 0,
		up[0], up[1], up[2], 0,
		look[0], look[1], look[2], 0,
		0., 0., 0., 1,
	}
}
*/
func (m Matrix4) MulS(scalar float32) Matrix4 {
	return Matrix4{
		m[0] * scalar, m[1] * scalar, m[2] * scalar, m[3] * scalar,
		m[4] * scalar, m[5] * scalar, m[6] * scalar, m[7] * scalar,
		m[8] * scalar, m[9] * scalar, m[10] * scalar, m[11] * scalar,
		m[12] * scalar, m[13] * scalar, m[14] * scalar, m[15] * scalar,
	}
}

func (m Matrix4) IMulS(scalar float32) {
	m[0] *= scalar
	m[1] *= scalar
	m[2] *= scalar
	m[3] *= scalar
	m[4] *= scalar
	m[5] *= scalar
	m[6] *= scalar
	m[7] *= scalar
	m[8] *= scalar
	m[9] *= scalar
	m[10] *= scalar
	m[11] *= scalar
	m[12] *= scalar
	m[13] *= scalar
	m[14] *= scalar
	m[15] *= scalar
}

// Mutiply this matrix with a column vector v, resulting in another column vector
func (m Matrix4) MulV(v Vector4) Vector4 {
	return Vector4{
		m[0]*v[0] + m[4]*v[1] + m[8]*v[2] + m[12]*v[3],
		m[1]*v[0] + m[5]*v[1] + m[9]*v[2] + m[13]*v[3],
		m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
		m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3],
	}
}

func (m Matrix4) DivS(scalar float32) Matrix4 {
	s := 1.0 / scalar
	return Matrix4{
		m[0] * s, m[1] * s, m[2] * s, m[3] * s,
		m[4] * s, m[5] * s, m[6] * s, m[7] * s,
		m[8] * s, m[9] * s, m[10] * s, m[11] * s,
		m[12] * s, m[13] * s, m[14] * s, m[15] * s,
	}
}

func (m Matrix4) Plus(q Matrix4) Matrix4 {
	return Matrix4{
		m[0] + q[0], m[1] + q[1], m[2] + q[2], m[3] + q[3],
		m[4] + q[4], m[5] + q[5], m[6] + q[6], m[7] + q[7],
		m[8] + q[8], m[9] + q[9], m[10] + q[10], m[11] + q[11],
		m[12] + q[12], m[13] + q[13], m[14] + q[14], m[15] + q[15],
	}
}

func (m Matrix4) IPlus(q Matrix4) {
	m[0] += q[0]
	m[1] += q[1]
	m[2] += q[2]
	m[3] += q[3]
	m[4] += q[4]
	m[5] += q[5]
	m[6] += q[6]
	m[7] += q[7]
	m[8] += q[8]
	m[9] += q[9]
	m[10] += q[10]
	m[11] += q[11]
	m[12] += q[12]
	m[13] += q[13]
	m[14] += q[14]
	m[15] += q[15]
}

func (m Matrix4) Sub(q Matrix4) Matrix4 {
	return Matrix4{
		m[0] - q[0], m[1] - q[1], m[2] - q[2], m[3] - q[3],
		m[4] - q[4], m[5] - q[5], m[6] - q[6], m[7] - q[7],
		m[8] - q[8], m[9] - q[9], m[10] - q[10], m[11] - q[11],
		m[12] - q[12], m[13] - q[13], m[14] - q[14], m[15] - q[15],
	}
}

func (m Matrix4) ISub(q Matrix4) {
	m[0] -= q[0]
	m[1] -= q[1]
	m[2] -= q[2]
	m[3] -= q[3]
	m[4] -= q[4]
	m[5] -= q[5]
	m[6] -= q[6]
	m[7] -= q[7]
	m[8] -= q[8]
	m[9] -= q[9]
	m[10] -= q[10]
	m[11] -= q[11]
	m[12] -= q[12]
	m[13] -= q[13]
	m[14] -= q[14]
	m[15] -= q[15]
}

func (m Matrix4) Identity() Matrix4 {
	m[0], m[1], m[2], m[3] = 1, 0, 0, 0
	m[4], m[5], m[6], m[7] = 0, 1, 0, 0
	m[8], m[9], m[10], m[11] = 0, 0, 1, 0
	m[12], m[13], m[14], m[15] = 0, 0, 0, 1
	return m
}

// Return the determinant
func (m Matrix4) Det() float32 {
	a0 := m[0]*m[5] - m[4]*m[1]
	a1 := m[0]*m[9] - m[8]*m[1]
	a2 := m[0]*m[13] - m[12]*m[1]
	a3 := m[4]*m[9] - m[8]*m[5]
	a4 := m[4]*m[13] - m[12]*m[5]
	a5 := m[8]*m[13] - m[12]*m[9]
	b0 := m[2]*m[7] - m[6]*m[3]
	b1 := m[2]*m[11] - m[10]*m[3]
	b2 := m[2]*m[15] - m[14]*m[3]
	b3 := m[6]*m[11] - m[10]*m[7]
	b4 := m[6]*m[15] - m[14]*m[7]
	b5 := m[10]*m[15] - m[14]*m[11]
	return a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0
}

func (m Matrix4) MustInverse() Matrix4 {
	if r, err := m.Inverse(); err == nil {
		return r
	}
	panic("determinant is zero")
}

func (m Matrix4) Inverse() (Matrix4, error) {
	a0 := m[0]*m[5] - m[4]*m[1]
	a1 := m[0]*m[9] - m[8]*m[1]
	a2 := m[0]*m[13] - m[12]*m[1]
	a3 := m[4]*m[9] - m[8]*m[5]
	a4 := m[4]*m[13] - m[12]*m[5]
	a5 := m[8]*m[13] - m[12]*m[9]
	b0 := m[2]*m[7] - m[6]*m[3]
	b1 := m[2]*m[11] - m[10]*m[3]
	b2 := m[2]*m[15] - m[14]*m[3]
	b3 := m[6]*m[11] - m[10]*m[7]
	b4 := m[6]*m[15] - m[14]*m[7]
	b5 := m[10]*m[15] - m[14]*m[11]
	det := a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0
	if Fabsf(det) <= internalε {
		return Matrix4{}, errors.New("determinant is zero")
	}
	id := 1. / det
	return Matrix4{
		id * (+m[5]*b5 - m[9]*b4 + m[13]*b3),
		id * (-m[1]*b5 + m[9]*b2 - m[13]*b1),
		id * (+m[1]*b4 - m[5]*b2 + m[13]*b0),
		id * (-m[1]*b3 + m[5]*b1 - m[9]*b0),
		id * (-m[4]*b5 + m[8]*b4 - m[12]*b3),
		id * (+m[0]*b5 - m[8]*b2 + m[12]*b1),
		id * (-m[0]*b4 + m[4]*b2 - m[12]*b0),
		id * (+m[0]*b3 - m[4]*b1 + m[8]*b0),
		id * (+m[7]*a5 - m[11]*a4 + m[15]*a3),
		id * (-m[3]*a5 + m[11]*a2 - m[15]*a1),
		id * (+m[3]*a4 - m[7]*a2 + m[15]*a0),
		id * (-m[3]*a3 + m[7]*a1 - m[11]*a0),
		id * (-m[6]*a5 + m[10]*a4 - m[14]*a3),
		id * (+m[2]*a5 - m[10]*a2 + m[14]*a1),
		id * (-m[2]*a4 + m[6]*a2 - m[14]*a0),
		id * (+m[2]*a3 - m[6]*a1 + m[10]*a0),
	}, nil
}

/*
// FIXME - fixme
func (m Matrix4) cofactor() Matrix4 {
	r := NewMatrix4()
	r[0] = (m[4]*m[8] - m[5]*m[7])
	r[1] = -(m[3]*m[8] - m[5]*m[6])
	r[2] = (m[3]*m[7] - m[4]*m[6])
	r[3] = -(m[1]*m[8] - m[2]*m[7])
	r[4] = (m[0]*m[8] - m[2]*m[6])
	r[5] = -(m[0]*m[7] - m[1]*m[6])
	r[6] = (m[1]*m[5] - m[2]*m[4])
	r[7] = -(m[0]*m[5] - m[2]*m[3])
	r[8] = (m[0]*m[4] - m[1]*m[3])
	return r
}



func (m Matrix4) NotEqual(q Matrix4) bool {
	return m != q
}

*/

func (m Matrix4) Equal(q Matrix4) bool {
	return m[0] == q[0] &&
		m[1] == q[1] &&
		m[2] == q[2] &&
		m[3] == q[3] &&
		m[4] == q[4] &&
		m[5] == q[5] &&
		m[6] == q[6] &&
		m[7] == q[7] &&
		m[8] == q[8] &&
		m[9] == q[9] &&
		m[10] == q[10] &&
		m[11] == q[11] &&
		m[12] == q[12] &&
		m[13] == q[13] &&
		m[14] == q[14] &&
		m[15] == q[15]
}

func (m Matrix4) Mul(q Matrix4) Matrix4 {
	r := NewMatrix4()
	r[0] = q[0]*m[0] + q[1]*m[4] + q[2]*m[8] + q[3]*m[12]
	r[1] = q[0]*m[1] + q[1]*m[5] + q[2]*m[9] + q[3]*m[13]
	r[2] = q[0]*m[2] + q[1]*m[6] + q[2]*m[10] + q[3]*m[14]
	r[3] = q[0]*m[3] + q[1]*m[7] + q[2]*m[11] + q[3]*m[15]
	r[4] = q[4]*m[0] + q[5]*m[4] + q[6]*m[8] + q[7]*m[12]
	r[5] = q[4]*m[1] + q[5]*m[5] + q[6]*m[9] + q[7]*m[13]
	r[6] = q[4]*m[2] + q[5]*m[6] + q[6]*m[10] + q[7]*m[14]
	r[7] = q[4]*m[3] + q[5]*m[7] + q[6]*m[11] + q[7]*m[15]
	r[8] = q[8]*m[0] + q[9]*m[4] + q[10]*m[8] + q[11]*m[12]
	r[9] = q[8]*m[1] + q[9]*m[5] + q[10]*m[9] + q[11]*m[13]
	r[10] = q[8]*m[2] + q[9]*m[6] + q[10]*m[10] + q[11]*m[14]
	r[11] = q[8]*m[3] + q[9]*m[7] + q[10]*m[11] + q[11]*m[15]
	r[12] = q[12]*m[0] + q[13]*m[4] + q[14]*m[8] + q[15]*m[12]
	r[13] = q[12]*m[1] + q[13]*m[5] + q[14]*m[9] + q[15]*m[13]
	r[14] = q[12]*m[2] + q[13]*m[6] + q[14]*m[10] + q[15]*m[14]
	r[15] = q[12]*m[3] + q[13]*m[7] + q[14]*m[11] + q[15]*m[15]
	return r
}

func (m Matrix4) IMul(q Matrix4) {
	temp := m.Mul(q)
	copy(m, temp)
}

// Transposed will *not* modify m
func (m Matrix4) Transpose() Matrix4 {
	return Matrix4{
		m[0], m[4], m[8], m[12],
		m[1], m[5], m[9], m[13],
		m[2], m[6], m[10], m[14],
		m[3], m[7], m[11], m[15],
	}
}

// ITranspose will modify m
func (m Matrix4) ITranspose() Matrix4 {
	temp := m.Transpose()
	copy(m, temp)
	return m
}

/*
Tests to see if the difference between two matrices,
element-wise, exceeds ε.
*/
func (a Matrix4) ApproxEquals(b Matrix4, ε float32) bool {
	for i := 0; i < 16; i++ {
		if delta := Fabsf(a[i] - b[i]); delta > ε {
			//fmt.Printf("delta between %f and %f is %f. ε=%f\m",a[i],b[i],delta,ε)
			return false
		}
	}
	return true
}

/*
// FIXME Orthogonalize will modify this matrix
func (m Matrix4) Orthogonalize(){
	i := NewVf(m[0],m[1],m[2])
	j := NewVf(m[3],m[4],m[5])
	k := NewVf(m[6],m[7],m[8]).Normalize();
	i = j.Cross(k).Normalize()
	j=k.Cross(i);
	m[0]=i[0]; m[3]=j[0]; m[6]=k[0]
	m[1]=i[3]; m[4]=j[3]; m[7]=k[3]
	m[2]=i[6]; m[5]=j[6]; m[8]=k[6]
}

// FIXME Orthogonalize will not modify this matrix
func (m1 Matrix4) Orthogonalized() Matrix4{
	m := m1.Copy()
	m.Orthogonalize();
	return m;
}

*/

//Returns the element at row,col
func (m Matrix4) at(row, col int) float32 {
	return m[row+col*4]
}

func (m Matrix4) Quaternion() Quaternion {
	// Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
	// article "HQuaternion Calculus and Fast Animation".
	toQuaternionNext := []int{1, 2, 0}

	q := Quaternion{0, 0, 0, 0}
	//fmt.Println("q = ", q)
	trace := m[0] + m[5] + m[10]
	var root float32
	//fmt.Printf("trace = %f\m", trace)
	if trace > 0. {
		// |w| > 1/2, may as well choose w > 1/2
		root = Sqrtf(trace + 1.0) // 2w
		q[0] = 0.5 * root
		root = 0.5 / root // 1/(4w)
		q[1] = (m.at(2, 1) - m.at(1, 2)) * root
		q[2] = (m.at(0, 2) - m.at(2, 0)) * root
		q[3] = (m.at(1, 0) - m.at(0, 1)) * root
	} else {
		// |w| <= 1/2
		i := 0
		if m.at(1, 1) > m.at(0, 0) {
			i = 1
		}
		if m.at(2, 2) > m.at(i, i) {
			i = 2
		}
		j := toQuaternionNext[i]
		k := toQuaternionNext[j]

		root = Sqrtf(m.at(i, i) - m.at(j, j) - m.at(k, k) + 1.)
		quat := q[1:]
		//fmt.Printf("Quat = [%f,%f,%f]\m", quat[0],quat[1],quat[2])
		quat[i] = 0.5 * root
		root = 0.5 / root
		q[0] = (m.at(k, j) - m.at(j, k)) * root
		quat[j] = (m.at(j, i) + m.at(i, j)) * root
		quat[k] = (m.at(k, i) + m.at(i, k)) * root
	}
	return q
}

func (m Matrix4) String() string {
	// output in octave format for easy testing
	return fmt.Sprintf("|%.5f,%.5f,%.5f,%.5f|\n|%.5f,%.5f,%.5f,%.5f|\n|%.5f,%.5f,%.5f,%.5f|\n|%.5f,%.5f,%.5f,%.5f|",
		m[0], m[4], m[8], m[12],
		m[1], m[5], m[9], m[13],
		m[2], m[6], m[10], m[14],
		m[3], m[7], m[11], m[15])
}

//Traslatioin matrix
func Trasl(x, y, z float32) Matrix4 {
	return Matrix4{
		1, 0, 0, x,
		0, 1, 0, y,
		0, 0, 1, z,
		0, 0, 0, 1,
	}
}

//Rotation matrices
func RotX(deg float32) Matrix4 {
	rad := deg * Deg2Rad
	rotm := NewMatrix4()
	rotm[5] = Cosf(rad)
	rotm[6] = -Sinf(rad)
	rotm[9] = -rotm[6]
	rotm[10] = rotm[5]
	return rotm
}

func RotY(deg float32) Matrix4 {
	rad := deg * Deg2Rad
	rotm := NewMatrix4()
	rotm[0] = Cosf(rad)
	rotm[2] = Sinf(rad)
	rotm[8] = -rotm[2]
	rotm[10] = rotm[0]
	return rotm
}

func RotZ(deg float32) Matrix4 {
	rad := deg * Deg2Rad
	rotm := NewMatrix4()
	rotm[0] = Cosf(rad)
	rotm[1] = Sinf(rad)
	rotm[4] = -rotm[1]
	rotm[5] = rotm[0]
	return rotm
}

//Scale matrix
func Scale(sx, sy, sz float32) Matrix4 {
	return Matrix4{
		sx, 0, 0, 0,
		0, sy, 0, 0,
		0, 0, sz, 0,
		0, 0, 0, 1,
	}
}

//set the gazing direction vector
func (m Matrix4) Z(dir Vector4) {
	m[8], m[9], m[10], m[11] = dir[0], dir[1], dir[2], dir[3]
}

//set the up direction vector
func (m Matrix4) Y(up Vector4) {
	m[4], m[5], m[6], m[7] = up[0], up[1], up[2], up[3]
}

//set the right direction vector
func (m Matrix4) X(right Vector4) {
	m[0], m[1], m[2], m[3] = right[0], right[1], right[2], right[3]
}

//set the position vector
func (m Matrix4) T(pos Vector4) {
	m[12], m[13], m[14], m[15] = pos[0], pos[1], pos[2], pos[3]
}

//get the gazing direction vector
func (m Matrix4) GetZ() Vector4 {
	return Vector4(m[8:12])
}

//get the up direction vector
func (m Matrix4) GetY() Vector4 {
	return Vector4(m[4:8])
}

//get the right direction vector
func (m Matrix4) GetX() Vector4 {
	return Vector4(m[0:4])
}

//get the position vector
func (m Matrix4) GetT() Vector4 {
	return Vector4(m[12:16])
}

//Move forward along the gazing direction (Z axis) of the camera, offset can be negative to reverse
func (m Matrix4) Forward(offset float32) {
	// row 3 += row 2 * offset
	step := m.GetZ().Mul(offset) // a step forward
	m.GetT().IAdd(step)          // take the step
}

//Move up along the up direction (Y axis) of the camera, offset can be negative to reverse
func (m Matrix4) Rise(offset float32) {
	// row 3 += row 2 * offset
	step := m.GetY().Mul(offset) // a step up
	m.GetT().IAdd(step)          // take the step
}

//Move right along the right direction (X axis) of the camera, offset can be negative to reverse
func (m Matrix4) Strafe(offset float32) {
	// row 3 += row 2 * offset
	step := m.GetX().Mul(offset) // a step right
	m.GetT().IAdd(step)          // take the step
}

//View() returns the view matrix to be multiplied to the ModelView matrix in OpenGl
//that is the inverse of the matrix obtained as explained here: http://content.gpwiki.org/index.php/MathGem:Fast_Matrix_Inversion
//WARNING: this works iff no scale is applied to the matrix, so no scaling please 
func (m Matrix4) View() Matrix4 {
	//T'X = -(RX dot T)
	T := m.GetT()
	tx := -m.GetX().Dot(T)
	ty := -m.GetY().Dot(T)
	tz := -m.GetZ().Dot(T)
	return Matrix4{
		m[0], m[4], m[8], 0.0,
		m[1], m[5], m[9], 0.0,
		m[2], m[6], m[10], 0.0,
		tx, ty, tz, 1,
	}
}

//Sets the node Point Of View (position) and point to Look At, both in vectors.
//LookAt is not the direction but the point been looked
func (m Matrix4) POWLookAt(pow, lookAt Vector4) (ret Matrix4) {
	forward := lookAt.Sub(pow).Normalize()
	//forward := pow.Sub(lookAt).Normalize()
	//right := Vector4{0.0, 1.0, 0.0, 0.0}.Cross(forward).Normalize()
	right := forward.Cross(Vector4{0.0, 1.0, 0.0, 0.0}).Normalize()
	//up := forward.Cross(right) //.Normalize()
	up := right.Cross(forward) //.Normalize()

	m[0] = right[0]
	m[1] = right[1]
	m[2] = right[2]
	m[3] = 0
	m[4] = up[0]
	m[5] = up[1]
	m[6] = up[2]
	m[7] = 0
	m[8] = -forward[0]
	m[9] = -forward[1]
	m[10] = -forward[2]
	m[11] = 0
	m[12] = pow[0]
	m[13] = pow[1]
	m[14] = pow[2]
	m[15] = 1
	return m
}
