#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/writeOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/adjacency_list.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>
#include <igl/massmatrix.h>
#include <igl/writeOBJ.h>
#include <fstream>
#include <cmath>
#include <array>
#include <ctime>
#include <stack>

const double M_PI = acos(double(-1));//3.1415926
const double theta = 0.1 / 180 * M_PI;//度 0.1
const double step = 0.1;

using namespace std;
using namespace Eigen;


/*
* clustering method
* DBSCAN
*/
class Point {
public:
    double x;
    double y;
    double z;
    int idx;//idx, normal vector index
    int cluster;//

    int pointType = 1;// 1 noise, 2 border, 3 core
    int pts = 0;//points in minPts;
    vector<int> corepts;//core points
    int visited = 0;

    Point() {};
    Point(double a, double b, double c, int d, int e) : x(a), y(b), z(c), idx(d), cluster(e) {};
};

double dotCalcuate(Point a, Point b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

std::vector<std::vector<int>> collectNeighbours(const std::vector<std::vector<int>>& adj,
    const Eigen::MatrixXd& V,//barycenter
    const Eigen::MatrixXd& N,//normal
    const double r,
    const double nr);//state the method

//DataBase Scan Algorithm
std::vector<std::vector<int>> collectNeighbours_DBSCAN(const std::vector<std::vector<int>>& adj, 
    const Eigen::MatrixXd& V,//barycenter
    const Eigen::MatrixXd& N,//normal
    const double r,//radius
    const double nr,//angle === eps, the first para
    int minPts = 3) //minPts--points number, the second para
{
    int len = V.rows();
    std::vector<std::vector<int>> result(len);//返回每一个面所相邻.收集到的所有面索引(类似双滤波情况) neighborhood collection
    result = collectNeighbours(adj, V, N, r, nr);
    const double normalConeThreshold = cos(nr * M_PI / 180.);//angle

    std::vector<Point> dataSet(len);
    for (int i = 0; i < len; ++i) {
        Point p(N(i, 0), N(i, 1), N(i, 2), i, i + 1);
        dataSet[i] = p;
    }

    //calculate pts;
    for (int i = 0; i < len; ++i) {
        for (int j = i + 1; j < len; j++) {
            if (dotCalcuate(dataSet[i], dataSet[j]) > normalConeThreshold) {//geo distance
                dataSet[i].pts++;
                dataSet[j].pts++;
            }
        }
    }
    
    std::vector<Point> corePoint;
    //core points
    for (int i = 0; i < len; ++i) {
        if (dataSet[i].pts >= minPts) {
            dataSet[i].pointType = 3;
            corePoint.emplace_back(dataSet[i]);
        }
    }
    //joint core point
    for (int i = 0; i < corePoint.size(); ++i) {
        for (int j = i + 1; j < corePoint.size(); ++j) {
            if (dotCalcuate(corePoint[i], corePoint[j]) > normalConeThreshold) {
                corePoint[i].corepts.emplace_back(j);
                corePoint[j].corepts.emplace_back(i);
            }
        }
    }

    //cluster mark
    for (int i = 0; i < corePoint.size(); ++i) {
        stack<Point*> st;//DFS
        if (corePoint[i].visited == 1) continue;
        st.push(&corePoint[i]);
        Point* p;
        while (!st.empty()) {  
            p = st.top();
            p->visited = 1;
            st.pop();
            for (int j = 0; j < p->corepts.size(); ++j) {
                if (corePoint[p->corepts[j]].visited == 1) continue;
                corePoint[p->corepts[j]].cluster = corePoint[i].cluster;
                corePoint[p->corepts[j]].visited = 1;
                st.push(&corePoint[p->corepts[j]]);
            }
        }
    }

    //border point, joint border point to core point
    for (int i = 0; i < len; ++i) {
        if (dataSet[i].pointType == 3) continue;
        for (int j = 0; j < corePoint.size(); ++j) {
            if (dotCalcuate(dataSet[i], corePoint[j]) > normalConeThreshold) {
                dataSet[i].pointType = 2;
                dataSet[i].cluster = corePoint[j].cluster;
                break;
            }
        }
    }

    //result; add the distance,(V.row(i) - V.row(corePoint[j].idx)).norm() < r 
    std::unordered_map<int, std::vector<int>> map;//cluster -- vector(result)
    for (int i = 0; i < len; ++i) {
        if (dataSet[i].pointType == 2) {
            int tmp = dataSet[i].cluster;
            map[tmp].emplace_back(dataSet[i].idx);
        }
    }

    for (int i = 0; i < corePoint.size(); ++i) {
        int tmp = corePoint[i].cluster;
        map[tmp].emplace_back(corePoint[i].idx);
    }

    /*for (std::pair<int, std::vector<int>> tmp : map) {
        for (int i : tmp.second) {
            cout << i << " ";
        }
        cout << endl;
    }*/

    //initial result,
    //for (int i = 0; i < len; ++i) {//pointType = 1
    //    result[i] = {i};
    //}
    //pointType = 2
    for (int i = 0; i < len; ++i) {
        int tmp = dataSet[i].cluster;
        if (dataSet[i].pointType == 2 && map.find(tmp) != map.end()) {
            //将DBCSAN结果和贪心结果进行交集
            vector<int> vectmp1 = map[tmp];
            vector<int> vectmp2 = result[dataSet[i].idx];
            sort(vectmp1.begin(), vectmp1.end());
            sort(vectmp2.begin(), vectmp2.end());
            
            vector<int> res;
            set_intersection(vectmp1.begin(), vectmp1.end(), vectmp2.begin(), vectmp2.end(), inserter(res, res.begin()));

            result[dataSet[i].idx] = res;
        }
    }
    //pointType = 3;
    for (int i = 0; i < corePoint.size(); ++i) {
        int tmp = corePoint[i].cluster;

        //将DBCSAN结果和贪心结果进行交集
        vector<int> vectmp1 = map[tmp];
        vector<int> vectmp2 = result[corePoint[i].idx];
        sort(vectmp1.begin(), vectmp1.end());
        sort(vectmp2.begin(), vectmp2.end());

        vector<int> res;
        set_intersection(vectmp1.begin(), vectmp1.end(), vectmp2.begin(), vectmp2.end(), inserter(res, res.begin()));
        result[corePoint[i].idx] = res;
    }

    return result;
}

/*
* ------------------------------------------
* ------------------------------------------
* ------------------------------------------
*/
void findRotations(const Eigen::MatrixXd& N0,
    const Eigen::MatrixXd& N1,
    std::vector<Eigen::Matrix3d>& rot) {// return rotation matrix

    const auto n = N0.rows();
    rot.resize(n);

    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d n1 = N0.row(i);
        Eigen::Vector3d n2 = N1.row(i);
        Eigen::Vector3d v = n1.cross(n2);
        const double c = n1.dot(n2);

        if (c > -1 + 1e-8) {
            const double coeff = 1 / (1 + c);
            Eigen::Matrix3d v_x;
            v_x << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
            rot[i] = Eigen::Matrix3d::Identity() + v_x + coeff * v_x * v_x;
        }
        else {
            rot[i] = -Eigen::Matrix3d::Identity();
        }
    }
}

std::vector<std::vector<int>> collectNeighbours(const std::vector<std::vector<int>>& adj,
    const Eigen::MatrixXd& V,//barycenter
    const Eigen::MatrixXd& N,//normal
    const double r,
    const double nr) {
    //depth first search
    std::vector<int> stack;//遍历过的面
    std::vector<int> flag(V.rows(), -1);//避免重复
    std::vector<std::vector<int>> result(V.rows());//通过栈，返回每一个面所相邻.收集到的所有面索引(类似双滤波情况) neighborhood collection
    const double normalConeThreshold = cos(nr * M_PI / 180.);//25

    for (int i = 0; i < V.rows(); ++i) {//质心位置

        stack.push_back(i);
        flag[i] = i;

        while (!stack.empty()) {
            auto id = stack.back();
            stack.pop_back();

            result[i].push_back(id);

            for (int j : adj[id]) {
                if (flag[j] != i && (V.row(i) - V.row(j)).norm() < r && (N.row(i).dot(N.row(j))) > normalConeThreshold) {
                    stack.push_back(j);
                    flag[j] = i;
                }
            }
        }
    }

    return result;
}

void fitNormals(const std::vector<std::vector<int>>& nbh,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& N,
    Eigen::MatrixXd& N2,
    const double cosineThreshold,
    const double sigma = 1.) {//fitting to update normals

    const auto nv = nbh.size();
    N2.resize(nv, 3);
    double angleThreshold = cosineThreshold * M_PI / 180.;

    for (int i = 0; i < nv; ++i) {

        const auto& nbi = nbh[i];

        Eigen::MatrixXd NN(nbi.size(), 3);

        for (int k = 0; k < nbi.size(); ++k) {
            NN.row(k) = N.row(nbi[k]);
        }

        Eigen::DiagonalMatrix<double, -1> W(nbi.size());

        if (sigma < 10.) {
            for (int i = 0; i < W.diagonal().size(); ++i) {
                double dot = NN.row(0).dot(NN.row(i));
                if (dot >= 1.) {
                    W.diagonal()(i) = 1;
                }
                else if (dot < 0) {
                    W.diagonal()(i) = 0;
                }
                else {
                    W.diagonal()(i) = std::exp(-std::pow(acos(dot) / angleThreshold / sigma, 2));
                }
            }
        }
        else {
            W.diagonal().setOnes();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(NN.transpose() * W * NN, Eigen::ComputeFullV);//Af
        Eigen::Matrix3d frame = svd.matrixV();//Xf
        Eigen::VectorXd weight = svd.singularValues();

        double thigma1 = weight[0];
        double thigma2 = weight[1];
        double thigma3 = weight[2];
        //std::cout << thigma1 << " " << thigma2 << " " << thigma3 << std::endl;
        double sum = thigma1 + thigma2 + thigma3;

        //PCA plane, 用Xf的特征向量张成的平面，得到target normal
        //if (thigma1 / sum > 0.99) {//thigma2可以忽略, 
        //    N2.row(i) = (frame.leftCols(1) * frame.leftCols(1).transpose() * N.row(i).transpose()).normalized();
        //}
        //else {
        //    N2.row(i) = (frame.leftCols(2) * frame.leftCols(2).transpose() * N.row(i).transpose()).normalized();
        //}

        N2.row(i) = (frame.leftCols(2) * frame.leftCols(2).transpose() * N.row(i).transpose()).normalized();
        //N2.row(i) = (frame.leftCols(1) * frame.leftCols(1).transpose() * N.row(i).transpose()).normalized();
    }
}

void assembleRHS(const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Eigen::Matrix3d>& R,
    Eigen::MatrixXd& rhs) {//right hand side

    const auto nv = V.rows();
    rhs.resize(nv, 3);
    rhs.setZero();

    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v0 = F(i, (j + 1) % 3);
            int v1 = F(i, (j + 2) % 3);

            Eigen::Vector3d b = C(i, j) * R[i] * (V.row(v0) - V.row(v1)).transpose();
            rhs.row(v0) -= b.transpose();
            rhs.row(v1) += b.transpose();
        }
    }
}

std::vector<std::vector<int>> triangleAdjacency(const Eigen::MatrixXi& F, const size_t nv) {

    std::vector<std::vector<int>> vnbhs(nv);//vertex neighborhoods, 点的索引，所对应的面索引
    const auto nf = F.rows();

    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < 3; ++j) {
            vnbhs[F(i, j)].push_back(i);
        }
    }

    std::vector<int> flags(nf, -1);//避免重复
    std::vector<std::vector<int>> ret(nf);//面索引所对应的相邻的面索引

    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k : vnbhs[F(i, j)]) {
                if (k != i && flags[k] != i) {
                    ret[i].push_back(k);
                    flags[k] = i;
                }
            }
        }
    }

    return ret;
}

void center(Eigen::MatrixXd& V) {
    V.rowwise() -= V.colwise().mean();//row wise逐行
    V /= 2. * V.rowwise().norm().maxCoeff();
}

void averageNormal(const Eigen::MatrixXd& N, Eigen::MatrixXd& N2)
{//test function and normalization on row 
    N2 = (N + N2) / 2.0;
    for (int i = 0; i < N2.rows(); ++i) {
        N2.row(i) = N2.row(i).normalized();
    }
}

void gaussThinning(const std::string& mesh_folder,
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& V,
    const int number_iterations = 100,
    double minConeAngle = 2.5,
    double smooth = 1e-5,
    double start_angle = 25,
    double radius = 0.1,
    double sigma = 2.) {

    double coneAngle = start_angle;
    double r = radius;
    double eps = 1e-3;

    V = V_in;
    const auto nv = V.rows();
    center(V);//normalized

    igl::writeOFF(mesh_folder + "/normalized.off", V, F);

    Eigen::SparseMatrix<double> I(nv, nv);
    I.setIdentity();

    Eigen::MatrixXi TT;
    Eigen::MatrixXd B, b, C, N, N2;
    std::vector<Eigen::Matrix3d> rot;
    Eigen::SparseMatrix<double> L, M;
    std::vector<std::vector<int>> nbhs;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;

    auto tt = triangleAdjacency(F, nv);
    igl::triangle_triangle_adjacency(F, TT);//库中方法, can be ignore
    igl::cotmatrix_entries(V, F, C);//每个角的余切值，return F.rows() * 3
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

    Eigen::MatrixXd A;
    A = -L + smooth * L.transpose() * L + eps * M;

    if (smooth) {
        chol.compute(-L + smooth * L.transpose() * L + eps * M);
    }
    else {
        chol.compute(-L + eps * M);
    }

    //test: change the target normal(average the souce normal and the target normal)

    std::vector<Eigen::Matrix3d> rot3;


    for (int k = 0; k < number_iterations; ++k) {

        igl::per_face_normals(V, F, N);
        igl::barycenter(V, F, B);

        //nbhs = collectNeighbours(tt, B, N, r, coneAngle);//面索引
        nbhs = collectNeighbours_DBSCAN(tt, B, N, r, coneAngle);//DBSCAN

        if (coneAngle > minConeAngle) coneAngle *= .95;

        fitNormals(nbhs, V, N, N2, coneAngle, sigma);
        //averageNormal(N, N2);//test function
        findRotations(N, N2, rot);//得到面法向量之间的旋转矩阵
        assembleRHS(C, V, F, rot, b);//装配B矩阵

        
        
        //get the gradient
        Eigen::MatrixXd grad;
        grad.resize(N.rows(), 3);

        //diff1 = A.inverse() * b + (eps * A.inverse() * M - Eigen::MatrixXd::Identity(A.rows(), A.rows())) * V;//D(nf`)
        Eigen::MatrixXd V1;
        V1 = chol.solve(eps * M * V - b);//求解稀疏线性方程组
        V1 = V1 - V;//diff1
        double diff1 = 0.;
        for (int i = 0; i < V1.rows(); ++i) {
            diff1 += V1.row(i).norm();
        }
        diff1 /= V1.rows();

        //diff2
        //采用罗德里格旋转公式，给定K轴的单位向量，使得旋转theta角度, 分别计算
        for (int i = 0; i < N.rows(); ++i) {
            for (int j = 0; j < 3; j++) {
                Eigen::MatrixXd N3, b3, V2;
                std::vector<Eigen::Matrix3d> rot3;
                N3 = N2;
                Eigen::Vector3d axis;
                if (j == 0) {
                    axis = { theta, 0, 0 };
                }
                else if (j == 1) {
                    axis = { 0, theta, 0 };
                }
                else {//j == 2
                    axis = { 0, 0, theta };
                }

                //Eigen::Vector3d v = N2.row(i).transpose();
                //N3.row(i) = (cos(theta) * v + axis.cross(v) * sin(theta) + axis.dot(v) * (1 - cos(theta)) * axis).transpose().normalized();//罗德里格旋转公式
                N3.row(i) = ((Eigen::Vector3d)N2.row(i) + axis).transpose().normalized();

                findRotations(N, N3, rot3);
                assembleRHS(C, V, F, rot3, b3);
                V2 = chol.solve(eps * M * V - b3);
                V2 = V2 - V;//diff2
                double diff2 = 0.;
                for (int i = 0; i < V2.rows(); ++i) {
                    diff2 += V2.row(i).norm();
                }
                diff2 /= V2.rows();

                grad(i, j) = (diff2 - diff1) / theta;
            }
        }

        
        //for (int i = 0; i < 10 && std::abs((N2 - last).maxCoeff()) > 1e-3; ++i) {
        double energy = 1.;
        for (int count = 0; count < 10 && energy > 1e-4; ++count) {

            findRotations(N, N2, rot);
            assembleRHS(C, V, F, rot, b);
            Eigen::MatrixXd V_tmp;
            V_tmp = chol.solve(eps * M * V - b);
            V_tmp -= V;
            energy = 0.;
            for (int i = 0; i < V_tmp.rows(); ++i) {
                energy += V_tmp.row(i).norm();
            }
            energy /= V_tmp.rows();

            cout << "energy-" << count << ": " << energy << endl;//判断是否能量下降

            N2 -= step * grad;//梯度下降
        }
        
        
        //梯度下降完成后，得到最后的顶点坐标

        findRotations(N, N2, rot);
        assembleRHS(C, V, F, rot, b);
        V = chol.solve(eps * M * V - b);

        //if (k % std::max(1, (number_iterations / 10)) == 0) {
        //    std::cout << "writing " + mesh_folder + ": " << k << "\n";
        //    igl::writeOFF(mesh_folder + "/out" + std::to_string(k) + ".off", V, F);
        //}
    }

    //std::cout << "gradient success!" << std::endl;
    return;
}

void runExperiment(std::string folder, std::string inputFile, std::string outputFile, const int iters, const double minAngle, const double start_angle = 25, const double radius = 0.1, const double smooth = 1e-5) {
    Eigen::MatrixXd V_in, V_out;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(folder + "/" + inputFile, V_in, F);
    gaussThinning(folder, V_in, F, V_out, iters, minAngle, smooth, start_angle, radius);
    igl::write_triangle_mesh(folder + "/" + outputFile, V_out, F);
}

int main(int argc, const char* argv[]) {

    clock_t start = clock();
    if (argc < 7) {
        std::cout << "Need input file, output file, output directory, number of iterations and minimum search cone. Running default experiments..." << std::endl;

        /* run default experiments here .... */
        runExperiment("..\\..\\examples\\architecture\\", "input.off", "out.off", 100, 2.5);
    }
    else
    {
        std::string  infile = argv[1];
        std::string  outfile = argv[2];
        std::string  folder = argv[3];

        auto numIters = std::atoi(argv[4]);
        auto minAngle = std::stold(argv[5]);
        auto start_angle = std::stold(argv[6]);

        std::cout << "Processing " << infile << " with " << numIters << " iterations, mimimum cone angle " << minAngle  << " and start angle " << start_angle << ". Output directory is " << folder << std::endl;

        runExperiment(folder, infile, outfile, numIters, minAngle, start_angle);
    }

    clock_t end = clock();
    double gaptime = (double)(end - start) / CLOCKS_PER_SEC;//seconds
    std::cout << "total time: " << gaptime << std::endl;
    return 0;
}
