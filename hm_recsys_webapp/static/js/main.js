let currentCustomerId = '';
let currentPage = 1;
const itemsPerPage = 18;
let currentFeed = [];
let currentResults = [];

document.addEventListener('DOMContentLoaded', function() {
    loadCustomers();
    loadFilterOptions();

    document.getElementById('customerSelect').addEventListener('change', function() {
        currentCustomerId = this.value;
        if (currentCustomerId) {
            getRecommendations(currentCustomerId);
        }
    });

    document.getElementById('filterContainer').addEventListener('change', function(e) {
        if (e.target.classList.contains('filter-select')) {
            if (currentCustomerId) {
                getRecommendations(currentCustomerId);
            }
        }
    });


});

function loadCustomers() {
    fetch('/get_customers')
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                const select = document.getElementById('customerSelect');
                select.innerHTML = '<option value="">Select a customer...</option>';
                data.forEach(id => {
                    const option = document.createElement('option');
                    option.value = id;
                    option.textContent = id.substring(0, 10) + '...';
                    select.appendChild(option);
                });
                if (data.length > 0) {
                    currentCustomerId = data[0];
                    select.value = data[0];
                    getRecommendations(data[0]);
                }
            } else {
                alert('Error loading customers: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => console.error('Error loading customers:', error));
}

function getRecommendations(customerId) {
    currentPage = 1;
    const filters = getSelectedFilters();
    const queryParams = new URLSearchParams(filters).toString();

    fetch(`/get_recommendations/${customerId}?${queryParams}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            currentFeed = data;
            displayFeed();
            document.getElementById('feedSection').classList.remove('d-none');
            document.getElementById('resultsSection').classList.add('d-none');
        })
        .catch(error => console.error('Error fetching recommendations:', error));
}

function loadFilterOptions() {
    fetch('/get_filter_options')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            const filterContainer = document.getElementById('filterContainer');
            filterContainer.innerHTML = '';
            for (const [key, values] of Object.entries(data)) {
                const col = document.createElement('div');
                col.className = 'col';
                const label = document.createElement('label');
                label.htmlFor = key;
                label.className = 'form-label filter-label';
                label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                
                const select = document.createElement('select');
                select.id = key;
                select.className = 'form-select filter-select';
                
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'All';
                select.appendChild(defaultOption);

                values.forEach(value => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = value;
                    select.appendChild(option);
                });
                col.appendChild(label);
                col.appendChild(select);
                filterContainer.appendChild(col);
            }
        })
        .catch(error => console.error('Error loading filters:', error));
}

function getSelectedFilters() {
    const filters = {};
    document.querySelectorAll('.filter-select').forEach(select => {
        if (select.value) {
            filters[select.id] = select.value;
        }
    });
    return filters;
}

function displayFeed() {
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageItems = currentFeed.slice(start, end);

    const grid = document.getElementById('feedGrid');
    grid.innerHTML = '';

    pageItems.forEach(item => {
        const col = document.createElement('div');
        col.className = 'col';
        col.innerHTML = `
            <div class="card h-100">
                <img src="${item.image_path}" class="card-img-top" alt="Product Image" onerror="this.src='https://via.placeholder.com/150'; this.alt='Image not found';">
                <div class="card-body d-flex flex-column">
                    <h4 class="card-title">${item.prod_name}</h4>
                    <p class="card-text"><small class="text-muted">ID: ${item.article_id}</small></p>
                    <p class="card-text mt-auto">${item.detail_desc}</p>
                </div>
            </div>
        `;
        grid.appendChild(col);
    });

    updatePagination('feedPagination', currentFeed.length, currentPage, 'feed');
}

function updatePagination(elementId, totalItems, currentPage, type) {
    const pagination = document.getElementById(elementId);
    pagination.innerHTML = '';
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    if (totalPages <= 1) return;

    const maxPagesToShow = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);
    if (endPage - startPage + 1 < maxPagesToShow) {
        startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }

    const liFirst = document.createElement('li');
    liFirst.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    liFirst.innerHTML = `<a class="page-link" href="#" onclick="changePage(1, '${type}'); return false;">First</a>`;
    pagination.appendChild(liFirst);

    const liPrev = document.createElement('li');
    liPrev.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    liPrev.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage - 1}, '${type}'); return false;">Prev</a>`;
    pagination.appendChild(liPrev);

    for (let i = startPage; i <= endPage; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${currentPage === i ? 'active' : ''}`;
        li.innerHTML = `<a class="page-link" href="#" onclick="changePage(${i}, '${type}'); return false;">${i}</a>`;
        pagination.appendChild(li);
    }

    const liNext = document.createElement('li');
    liNext.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    liNext.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage + 1}, '${type}'); return false;">Next</a>`;
    pagination.appendChild(liNext);

    const liLast = document.createElement('li');
    liLast.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    liLast.innerHTML = `<a class="page-link" href="#" onclick="changePage(${totalPages}, '${type}'); return false;">Last</a>`;
    pagination.appendChild(liLast);
}

function changePage(page, type) {
    const totalItems = type === 'feed' ? currentFeed.length : currentResults.length;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    if (page < 1 || page > totalPages) return;

    currentPage = page;
    if (type === 'feed') {
        displayFeed();
    } else {
        displayResults();
    }
}

function search(searchType) {
    if (!currentCustomerId) {
        alert('Please select a customer first.');
        return;
    }

    const formData = new FormData();
    formData.append('customer_id', currentCustomerId);
    formData.append('search_type', searchType);

    const filters = getSelectedFilters();
    for (const [key, value] of Object.entries(filters)) {
        formData.append(key, value);
    }

    if (searchType === 'text') {
        const query = document.getElementById('textQuery').value.trim();
        if (!query) {
            alert('Please enter a text query.');
            return;
        }
        formData.append('query', query);
    } else {
        const fileInput = document.getElementById('imageUpload');
        if (fileInput.files.length === 0) {
            alert('Please upload an image.');
            return;
        }
        formData.append('file', fileInput.files[0]);
    }

    fetch('/search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Search error: ' + data.error);
            return;
        }
        currentResults = data;
        currentPage = 1;
        document.getElementById('feedSection').classList.add('d-none');
        document.getElementById('resultsSection').classList.remove('d-none');
        displayResults();
    })
    .catch(error => console.error('Error:', error));
}

function displayResults() {
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageItems = currentResults.slice(start, end);

    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = '';

    pageItems.forEach(item => {
        const col = document.createElement('div');
        col.className = 'col';
        col.innerHTML = `
            <div class="card h-100">
                <img src="${item.image_path}" class="card-img-top" alt="Product Image" onerror="this.src='https://via.placeholder.com/150'; this.alt='Image not found';">
                <div class="card-body d-flex flex-column">
                    <h4 class="card-title">${item.prod_name}</h4>
                    <p class="card-text"><small class="text-muted">ID: ${item.article_id}</small></p>
                    <p class="card-text mt-auto">${item.detail_desc}</p>
                </div>
            </div>
        `;
        grid.appendChild(col);
    });

    updatePagination('resultsPagination', currentResults.length, currentPage, 'results');
}

function clearSearch() {
    document.getElementById('textQuery').value = '';
    document.getElementById('imageUpload').value = '';
    document.getElementById('resultsSection').classList.add('d-none');
    document.getElementById('feedSection').classList.remove('d-none');
    displayFeed();
}
