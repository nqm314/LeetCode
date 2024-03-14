#include <bits/stdc++.h>
using namespace std;

    class StockSpanner {
    public:
        vector<int> v;
        StockSpanner() {
        }
        
        stack<pair<int, int>> s;
        int next(int price) {
            int res = 1;
            while (!s.empty() && s.top().first <= price) {
                res += s.top().second;
                s.pop();
            }
            s.push({price, res});
            return res;
        }
    };

/**
 * Your StockSpanner object will be instantiated and called as such:
 * StockSpanner* obj = new StockSpanner();
 * int param_1 = obj->next(price);
 */

    int minCost1(string colors, vector<int>& neededTime) {
        int n = colors.size(), res = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (colors[i] == colors[i + 1]) {
                int sz = 1;
                priority_queue<int, vector<int>, greater<int>> pq;
                pq.push(neededTime[i]);
                while (colors[i] == colors[i + 1]) {
                    sz++;
                    pq.push(neededTime[i+1]);
                    ++i;
                }
                while (sz > 1) {
                    res += pq.top();
                    pq.pop();
                    sz--;
                }
            }
        }
        return res;
    }

    int minCost(string colors, vector<int>& neededTime) {
        int n = colors.size(), res = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (colors[i] == colors[i + 1]) {
                int sum_group = neededTime[i], max_group = neededTime[i];
                while (colors[i] == colors[i + 1]) {
                    max_group = max(max_group, neededTime[i + 1]);
                    sum_group += neededTime[i + 1];
                    ++i;
                }
                res += sum_group - max_group;
            }
        }
        return res;
    }

    int longestSubarray(vector<int>& nums) {
        int zeroCount = 0, longestWindow = 0, start = 0;
        for (int i = 0; i < nums.size(); ++i) {
            zeroCount += (nums[i] == 0);
            while (zeroCount > 1) {
                zeroCount -= (nums[start] == 0);
                start++;
            }
            longestWindow = max(longestWindow, i - start);
        }
        return longestWindow;
    }

 void combination3(vector<vector<int>>& res, vector<int>& sum, int k, int n) {
        if (sum.size() == k && n == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 1 : sum.back() + 1; i <= 9; ++i) {
            if ((n - i) < 0) break;
            sum.push_back(i);
            combination3(res, sum, k, n - i);
            sum.pop_back();
        }
}

    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> sum;
        combination3(res, sum, k, n);
        return res;
    }

    int getidx(vector<int>& candidates, int x) {
        for (int i = 0; i < candidates.size(); ++i) {
            if (candidates[i] == x) return i;
        }
        return -1;
    }

    void combination(vector<vector<int>>& res, vector<int>& sum, vector<int>& candidates, int target) {
        if (target == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 0 : getidx(candidates, sum.back()); i < candidates.size(); ++i) {
            if (target - candidates[i] < 0) break;
            sum.push_back(candidates[i]);
            combination(res, sum, candidates, target - candidates[i]);
            sum.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> sum;
        sort(candidates.begin(), candidates.end());
        combination(res, sum, candidates, target);
        return res;
    }

    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        stack<int> st;
        vector<int> res(n, 0);
        for (int i = 0; i < n; ++i) {
            if (st.empty()) st.push(i);
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                res[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);
        }
        while (!st.empty()) {
            res[st.top()] = 0;
            st.pop();
        }
        return res;
    }

    bool sieve[5000007] = {0};
    void init(int n) {
        for (int i = 2; i <= n; ++i ) {
            sieve[i] = true;
        }
        for (int i = 2; i*i <= n; ++i) {
            if (sieve[i]) {
                for (int j = i*i; j <= n; j += i) {
                    sieve[j] = false;
                }
            }
        }
    }

    int countPrimes(int n) {
        init(n);
        int cnt = 0;
        for (int i = 2; i < n; ++i) {
            if (sieve[i]) cnt++;
        }
        return cnt;
    }

    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int i = 0, j = 0, res = 0;
        while (i < g.size() && j < s.size()) {
            if (s[j] >= g[i]) {
                i++, j++, res++;
            }
            else {
                j++;
            }
        }
        return res;
    }

    void combination2(vector<vector<int>>& res, vector<int>& sum, vector<int>& sum_idx, vector<int>& candidates, int target, int start) {
        if (target == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 0 : sum_idx.back() + 1; i < candidates.size(); ++i) {
            if (target - candidates[i] < 0) break;
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            sum_idx.push_back(i);
            sum.push_back(candidates[i]);
            combination2(res, sum, sum_idx, candidates, target - candidates[i], i + 1);
            sum_idx.pop_back();
            sum.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> sum;
        vector<int> sum_idx;
        sort(candidates.begin(), candidates.end());
        combination2(res, sum, sum_idx, candidates, target, 0);
        return res;
    }

    vector<vector<int>> findMatrix(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size(); 
        int idx = 0;
        vector<bool> flag (n, false);
        while (idx < n) {
            vector<int> sub;
            vector<bool> visited (201, false);
            for (int i = 0; i < n; ++i) {
                if ((!flag[i] && !visited[nums[i]])) {
                    sub.push_back(nums[i]);
                    idx++;
                    flag[i] = true;
                    visited[nums[i]] = true;
                }
            }
            res.push_back(sub);
            
        }
        return res;
    }

    bool fullZero(string s) {
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '1') return false;
        }
        return true;
    }

    int numberOfBeams(vector<string>& bank) {
        int n = bank.size();
        int res = 0;
        vector<int> helper;
        for (int i = 0; i < n; ++i) {
            if (fullZero(bank[i])) continue;
            int cnt = 0;
            for (char c : bank[i]) {
                if (c == '1') cnt++;
            }
            helper.push_back(cnt);
        }
        if (helper.size() == 1 || helper.size() == 0) return 0;
        for (int i = 0; i < helper.size() - 1; ++i) {
            res += helper[i] * helper[i + 1];
        }
        return res;
    }

    int minOperations(vector<int>& nums) {
        map<int, int> mp;
        for (int i = 0; i < nums.size(); ++i) {
            mp[nums[i]]++;
        }
        int res = 0;
        for (auto x : mp) {
            if (x.second == 1) return -1;
            res += ceil((double)x.second / 3);
        }
        return res;
    }

    int jump(vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0; // Already at the end

        int maxReach = nums[0]; // Farthest index you can reach from the current position
        int end = nums[0];      // End of the current jump
        int steps = 1;          // Number of jumps made so far

        for (int i = 1; i < n - 1; ++i) {
            maxReach = max(maxReach, i + nums[i]);
            
            if (i == end) { // If you reach the end of the current jump
                end = maxReach; // Update the end of the current jump
                steps++;        // Make a jump
            }
        }

        return steps;
    }

    // int jump(vector<int>& nums) {
    //     int n = nums.size();
    //     vector<int> dp(n, 0);
    //     for (int i = n - 2; i >= 0; --i){
    //         if (nums[i] == 0) {
    //             dp[i] = -1;
    //             continue;
    //         }
    //         for (int j = 1; j <= nums[i]; ++j) {
    //             if (i + j == n - 1) {
    //                 dp[i] = min(dp[i] + 1, dp[i + j] + 1);
    //                 break;
    //             }
    //             else {
    //                 if (dp[i + j] == -1) continue;
    //                 dp[i] = dp[i] == 0 ? dp[i + j] + 1 : min(dp[i], dp[i + j] + 1);
    //             }
    //         }
    //         dp[i] = dp[i] == 0 ? -1: dp[i];
    //     }
    //     return dp[0];
    // }

    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int reach = 0;
        for (int i = 0; i < n; ++i) {
            if (reach < i) return false;
            reach = max(reach, i + nums[i]);
        }
        return true;
    }

    int minPathSum(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        int dp[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) dp[i][j] = INT_MAX;
        }
        dp[0][0] = grid[0][0];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i + 1 < n) dp[i + 1][j] = min(dp[i][j] + grid[i + 1][j], dp[i + 1][j]);
                if (j + 1 < m) dp[i][j + 1] = min(dp[i][j] + grid[i][j + 1], dp[i][j + 1]);
            }
        }
        return dp[n - 1][m - 1];
    }

    bool visited[501][501];
    int x_axis[4] = {-1, 1, 0, 0}, y_axis[4] = {0,0,-1,1};
    void dfs(vector<vector<char>>& grid, int m, int n, int x, int y, pair<int,int> lastCell, bool &flag) {
        visited[x][y] = true;
        for (int k = 0; k < 4; k++) {
            int newx = x + x_axis[k], newy = y + y_axis[k];
            if (newx >= 0 && newy >= 0 && newx < m && newy < n) {
                if (newx != lastCell.first || newy != lastCell.second) {
                    if (grid[newx][newy] == grid[x][y]) {
                        if (visited[newx][newy]) {
                            flag = 1;
                            return;
                        }
                        pair<int,int> lastcell = make_pair(x, y);
                        dfs(grid, m, n, newx, newy, lastcell, flag);
                    } 
                }
            }
        }
    }

    bool containsCycle(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) visited[i][j] = false;
        }
        bool flag = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!visited[i][j]) {
                    pair<int, int> lastCell = {i, j};
                    dfs(grid, m, n, i, j, lastCell, flag);
                    if (flag) return true;
                }
            }
        }
        return false;
    }

    int minFallingPathSumHelper(vector<vector<int>>& matrix, int r, int c, vector<vector<int>>& dp){
        if(r == matrix.size()-1 and c < matrix[0].size() and c >= 0) return matrix[r][c]; 
        if(c >= matrix[0].size() or c < 0) return INT_MAX;
        
        if(dp[r][c] != INT_MAX) return dp[r][c];
        return dp[r][c] = matrix[r][c] + min(min(minFallingPathSumHelper(matrix, r+1, c-1, dp), minFallingPathSumHelper(matrix, r+1, c, dp)), minFallingPathSumHelper(matrix, r+1, c+1, dp));
        
    }
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int rows = matrix.size(), cols = matrix[0].size();
        vector<vector<int>> dp(rows+1, vector<int>(cols+1, INT_MAX));
        int ans = INT_MAX;
        for(int c=0; c < cols; c++){
            ans = min(ans, minFallingPathSumHelper(matrix, 0, c, dp));
        }
        return ans;
    }

    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };

    int helper(TreeNode* root) {
        if (!root) return INT_MAX;
        if (!root->left && !root->right) return 1;
        return 1 + min(helper(root->left), helper(root->right));
    }

    int minDepth(TreeNode* root) {
        if (!root) return 0;
        return helper(root);
    }

    int sumSubarrayMins(vector<int>& arr) {
        int n = arr.size();
        stack<pair<int, int>> stl, str;
        vector<int> left(n), right(n);

        for (int i = 0; i < n; i++) left[i] = i + 1;

        for (int i = 0; i < n; i++) right[i] = n - i;

        for (int i = 0; i < n; i++) {
            while (!stl.empty() && stl.top().first > arr[i]) stl.pop();
            left[i] = (stl.empty()) ? i + 1 : i - stl.top().second;
            stl.push(make_pair(arr[i], i));

            while (!str.empty() && str.top().first > arr[i]) {
                pair p = str.top(); str.pop();
                right[p.second] = i - p.second;
            }
            str.push(make_pair(arr[i], i));
        }

        int mod = 1e9 + 7;
        long long sum = 0;
        for (int i = 0; i < n; i++) {
            long long subsum = (((long long)arr[i] % mod) * (left[i] % mod) * (right[i] % mod)) % mod;
            sum = ((sum%mod) + subsum) % mod;
        }
        return (int)sum;
    }

    int sumSubarrayMaxs(vector<int>& arr) {
        int n = arr.size();
        stack<pair<int, int>> stl, str;
        vector<int> left(n), right(n);

        for (int i = 0; i < n; i++) left[i] = i + 1;

        for (int i = 0; i < n; i++) right[i] = n - i;

        for (int i = 0; i < n; i++) {
            while (!stl.empty() && stl.top().first < arr[i]) stl.pop();
            left[i] = (stl.empty()) ? i + 1 : i - stl.top().second;
            stl.push(make_pair(arr[i], i));

            while (!str.empty() && str.top().first < arr[i]) {
                pair p = str.top(); str.pop();
                right[p.second] = i - p.second;
            }
            str.push(make_pair(arr[i], i));
        }

        int mod = 1e9 + 7;
        long long sum = 0;
        for (int i = 0; i < n; i++) {
            long long subsum = (((long long)arr[i] % mod) * (left[i] % mod) * (right[i] % mod)) % mod;
            sum = ((sum%mod) + subsum) % mod;
        }
        return (int)sum;
    }

    long long subArrayRanges(vector<int>& nums) {
        return sumSubarrayMaxs(nums) - sumSubarrayMins(nums);
    }

    static bool comp(vector<int>& a, vector<int>& b) {
        return a[1] < b[1];
    }

    int findMinArrowShots(vector<vector<int>>& points) {
        int n = points.size(), res = 0;
        if (n == 1) return 1;
        sort(points.begin(), points.end(), comp);
        for (auto x : points){
            for (int y : x) {
                cout << y << " ";
            }
        }
        cout << endl;
        vector<int> prev = points[0];
        int i;
        for (i = 0; i < n - 1; i++) {
            if (prev[1] >= points[i + 1][0]) {
                while (i < n - 1 && prev[1] >= points[i + 1][0]) {
                    i++;
                }
                prev = i < n - 1 ? points[i + 1] : prev;
                res++;
            } 
            else {
                if (i == n - 2) {
                    res += 2;
                }
                else {
                    prev = points[i + 1];
                    res++;
                }
            }
        }
        return res + (i == (n - 1));
    }

    vector<int> sequentialDigits(int l, int h) {
        queue<int> q;
        for(int i = 1; i <= 9; i++) {
            q.push(i);
        }
        vector<int> ret;
        while(!q.empty()) {
            int f = q.front();
            q.pop();
            if(f <= h && f >= l) {
                ret.push_back(f);
            }
            if(f > h)
                break;
            int num = f % 10;
            if(num < 9) {
                q.push(f * 10 + (num + 1));
            }
        }
        return ret;
    }

    string minWindow(string s, string t) {
        vector<int> map(128,0);
        for(auto c: t) map[c]++;
        int counter=t.size(), begin=0, end=0, d=INT_MAX, head=0;
        while(end<s.size()){
            if(map[s[end++]]-->0) counter--; //in t
            while(counter==0){ //valid
                if(end-begin<d)  d=end-(head=begin);
                if(map[s[begin++]]++==0) counter++;  //make it invalid
            }  
        }
        return d==INT_MAX? "" :s.substr(head, d);
    }

    void dfs(string start, string end, map<string, double>&mp, map<string,vector<string>>& graph, double& val, map<string,int>& visited, bool& found) {
        visited[start] = 1;
        if (found == true) return;
        for (auto child : graph[start]) {
            if (visited[child] != 1) {
                val *= mp[start+"->"+child];
                if (end == child) {
                    found = true;
                    return;
                }
                dfs(child, end, mp, graph, val, visited, found);
                if (found) return;
                else val /= mp[start+"->"+child];
            }
        }
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        vector<double> ans;
        map<string,double> mp;
        map<string,vector<string>> graph;
        for (int i = 0; i < equations.size(); i++) {
            string u = equations[i][0], v = equations[i][1];
            mp[u+"->"+v] = values[i];
            mp[v+"->"+u] = 1/values[i];
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
        for (int i = 0; i < queries.size(); i++) {
            string start = queries[i][0], end = queries[i][1];
            if (graph.find(start) == graph.end() || graph.find(end) == graph.end()){
                ans.push_back(-1);
            }
            else {
                double val = 1;
                map<string, int> visited;
                bool found = false;
                if (start == end) found = true;
                else {
                    dfs(start, end, mp, graph, val, visited, found);
                }
                if (found) ans.push_back(val);
                else ans.push_back(-1);
            }
        }
        return ans;
    }

    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int n = nums.size(), max = 0, index = 0;
        vector<int> cnt(n), prev(n, -1);
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; i++) {
            cnt[i] = 1, prev[i] = -1;
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] % nums[j] == 0) {
                    if (cnt[i] < 1 + cnt[j]) {
                        cnt[i] = 1 + cnt[j];
                        prev[i] = j;
                    }
                }
            }
            if (cnt[i] > max) {
                max = cnt[i];
                index = i;
            }
        }
        vector<int> res;
        while (index != -1) {
            res.push_back(nums[index]);
            index = prev[index];
        }
        return res;
    }

    int furthestBuilding(vector<int>& A, int bricks, int ladders) {
        priority_queue<int> pq;
        for (int i = 0; i < A.size() - 1; i++) {
            int d = A[i + 1] - A[i];
            if (d > 0)
                pq.push(-d);
            if (pq.size() > ladders) {
                bricks += pq.top();
                pq.pop();
            }
            if (bricks < 0)
                return i;
        }
        return A.size() - 1;
    }

    int findJudge(int n, vector<vector<int>>& trust) {
        map<int, int> mp;
        for (int i = 0; i < trust.size(); ++i) {
            if (!mp.count(trust[i][1])) mp[trust[i][1]];
            else mp[trust[i][1]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (mp[i] == 0) return i;
        }
        return -1;        
    }

    
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        vector<vector<pair<int, int>>> adj(101);
        for (int i = 0; i < flights.size(); ++i) {
            adj[flights[i][0]].push_back(make_pair(flights[i][1], flights[i][2]));
        }
        vector<int> cost(n, INT_MAX);
        cost[src] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, src});
        while (!pq.empty()) {
            pair<int, int> top = pq.top();;
            pq.pop();
            int u = top.second, cst = top.first;
            if (cst > cost[u]) continue;
            for (auto it : adj[u]) {
                int v = it.first, wgt = it.second;
                if (cost[v] > cost[u] + wgt) {
                    cost[v] = cost[u] + wgt;
                    pq.push({cost[v], v});
                }
            }
        }
        return cost[dst];
    }

    vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
        sort(products.begin(), products.end());
        string prefix = "";
        int start, bsStart = 0, n = products.size();
        vector<vector<string>> res;
        for (char c : searchWord) {
            prefix += c;
            start = lower_bound(products.begin() + bsStart, products.end(), prefix) - products.begin();
            res.push_back({});
            for (int i = start; i < min(start + 3, n) && !products[i].compare(0, prefix.length(), prefix); ++i) {
                res.back().push_back(products[i]);
            }
        }
        return res;
    }

    int gcd(int a, int b) {
        if (a == 0) return b;
        if (b == 0) return a;
        return gcd(b, a%b);
    }

    bool bfs(int n, int u, int v, vector<vector<int>>& adj) {
        vector<bool> visited(100007, false);
        visited[u] = true;
        queue<int> q;
        q.push(u);
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            for (int y : adj[x]) {
                if (!visited[y]) {
                    if (y == v) return true;
                    visited[y] = true;
                    q.push(y);
                }
            }
        }
        return false;
    }

    bool canTraverseAllPairs(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> adj(n);
        for (int i = 0; i < nums.size() - 1; ++i) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (gcd(nums[i], nums[j]) > 1) {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }
        for (int i = 0; i < nums.size() - 1; ++i) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (!bfs(n, i, j, adj)) return false;
            }
        }
        return true;
    }


//Problem 1609. Even-odd tree
class problem1609 {
    public:
    bool isEvenOddTree(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        bool even = true;
        int prev;
        while (!q.empty()) {
            int sz = q.size();
            prev = even ? INT_MIN : INT_MAX;
            while (sz) {
                TreeNode* curr = q.front();
                q.pop();
                if (even) {
                    if (curr->val % 2 == 0 || curr->val <= prev) return false;
                }
                else {
                    if (curr->val % 2 == 1 || curr->val >= prev) return false;
                }
                prev = curr->val;
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
                sz--;
            }
            even = !even;
        }
        return true;
    }
};


//Problem 2864. Maximum Odd Binary Number 
class problem2864 {
    public:
    string maximumOddBinaryNumber(string s) {
        ios::sync_with_stdio(false);
        cin.tie(0);
        cout.tie(0);
        int n = s.size(), cnt = 0;
        for (char c : s) 
            if (c == '1') cnt++;
        string res = "";
        for (int i = 0; i < cnt - 1; ++i) res += '1';
        for (int i = 0; i < n - cnt; ++i) res += '0';
        return res + '1';
    }
};


//Problem 1579.Remove-max-number-of-edges-to-keep-graph-fully-traversable
class problem1579 {
    private:
    class UnionFind {
        vector<int> component;
        int distinctComponents;
        public:
            /*
            *   Initially all 'n' nodes are in different components.
            *   e.g. component[2] = 2 i.e. node 2 belong to component 2.
            */
            UnionFind(int n) {
                distinctComponents = n;
                for (int i=0; i<=n; i++) {
                    component.push_back(i);
                }
            }
            
            /*
            *   Returns true when two nodes 'a' and 'b' are initially in different
            *   components. Otherwise returns false.
            */
            bool unite(int a, int b) {       
                if (findComponent(a) == findComponent(b)) {
                    return false;
                }
                component[findComponent(a)] = b;
                distinctComponents--;
                return true;
            }
            
            /*
            *   Returns what component does the node 'a' belong to.
            */
            int findComponent(int a) {
                if (component[a] != a) {
                    component[a] = findComponent(component[a]);
                }
                return component[a];
            }
            
            /*
            *   Are all nodes united into a single component?
            */
            bool united() {
                return distinctComponents == 1;
            }
    };

    public:
// ----------------- Actual Solution --------------
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        // Sort edges by their type such that all type 3 edges will be at the beginning.
        sort(edges.begin(), edges.end(), [] (vector<int> &a, vector<int> &b) { return a[0] > b[0]; });
        
        int edgesAdded = 0; // Stores the number of edges added to the initial empty graph.
        
        UnionFind bob(n), alice(n); // Track whether bob and alice can traverse the entire graph,
                                    // are there still more than one distinct components, etc.
        
        for (auto &edge: edges) { // For each edge -
            int type = edge[0], one = edge[1], two = edge[2];
            switch(type) {
                case 3:
                    edgesAdded += (bob.unite(one, two) | alice.unite(one, two));
                    break;
                case 2:
                    edgesAdded += bob.unite(one, two);
                    break;
                case 1:
                    edgesAdded += alice.unite(one, two);
                    break;
            }
        }
        
        return (bob.united() && alice.united()) ? (edges.size()-edgesAdded) : -1; // Yay, solved.
    }
};
    

//  Problem 948.Bag of Tokens
class problem948 {
    public:
    int bagOfTokensScore(vector<int>& tokens, int power) {
        int score = 0;
        sort(tokens.begin(), tokens.end());
        int i = 0, j = tokens.size() - 1;
        while (i <= j) {
            if (power >= tokens[i]) {
                power -= tokens[i];
                score++;
                i++;
            }
            else if (i < j && score){
                power += tokens[j];
                score--;
                j--;
            }
            else break;
        }
        return score;
    }
};


//Problem 1750. Minimum-length-of-string-after-deleting-similar-ends  
class problem1750 {
    public:
    int minimumLength(string s) {
        int res = s.size();
        int i = 0, j = res - 1;
        while (i < j) {
            if (s[i] != s[j]) break;
            while (i < j && s[i] == s[j]) {
                res -= 2;
                ++i, --j;
            }
            while (i > 0 && i <= j && s[i] == s[i - 1]) {
                res--;
                ++i;
            }
            while (j < s.size() - 1 && i < j && s[j] == s[j + 1]) {
                res--;
                --j;
            }
        }
        return res;
    }
};


//problem 930. Binary Subarrays With Sum
class problem930 {
    private:
    //// Helper function to count the number of subarrays with sum at most the given goal
    int slidingWindowsAtMostGoal(vector<int>& nums, int goal) {
        int start = 0, currentSum = 0, totalCnt = 0;
        // Iterate through the array using a sliding window approach
        for (int end = 0; end < nums.size(); end++) {
            currentSum += nums[end];

            // Adjust the window by moving the start pointer to the right
            // until the sum becomes less than or equal to the goal
            while (start <= end && currentSum > goal) {
                currentSum -= nums[start++];
            }

            // Update the total count by adding the length of the current subarray
            totalCnt += end - start + 1;
        }
        return totalCnt;
    }

    public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return slidingWindowsAtMostGoal(nums, goal) - slidingWindowsAtMostGoal(nums, goal - 1);
    }
};
    
    void nam_moi_Giap_Thin() {
        cout << "Chuc mung nam moi Giap Thin" << endl;
        cout << "Thanh cong - Hoc gioi - Hoc bong - GPA > 3.6 - Co nguoi yeu - Tang chieu cao:))))" << endl;
    }

int main() {
    cout << "I'm Minh. I am studying Bachelors of Computer Science at Ho Chi Minh City University of Technology (HCMUT, VNU-HCM).";
}