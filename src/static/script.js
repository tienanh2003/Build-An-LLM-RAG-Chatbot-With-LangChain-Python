/**********************
 * DATA & STATE
 **********************/
let chatData = [];
let currentChatId = null;
let pendingImages = [];

/**********************
 * COOKIE: Lưu & Tải dữ liệu
 **********************/
function setCookie(name, value, days = 30) {
  const date = new Date();
  date.setTime(date.getTime() + (days*24*60*60*1000));
  const expires = "expires=" + date.toUTCString();
  document.cookie = name + "=" + encodeURIComponent(value) + ";" + expires + ";path=/";
}
function getCookie(name) {
  const decodedCookie = decodeURIComponent(document.cookie);
  const ca = decodedCookie.split(';');
  const prefix = name + "=";
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i].trim();
    if (c.indexOf(prefix) === 0) {
      return c.substring(prefix.length, c.length);
    }
  }
  return "";
}
function saveToCookie() {
  setCookie("chatData", JSON.stringify(chatData));
}
function loadFromCookie() {
  const cookieValue = getCookie("chatData");
  if (cookieValue) {
    try {
      chatData = JSON.parse(cookieValue);
    } catch (e) {
      chatData = [];
    }
  }
}

/**********************
 * CORE FUNCTIONS
 **********************/
function generateChatId() {
  return "chat_" + Date.now() + "_" + Math.floor(Math.random() * 10000);
}

function createNewConversation() {
  const newId = generateChatId();
  const newTitle = "Hội thoại mới";
  const timeNow = new Date().toLocaleTimeString("vi-VN", { hour: '2-digit', minute:'2-digit' });
  const newChat = {
    id: newId,
    title: newTitle,
    messages: [
      { role: "assistant", content: "Xin chào, tôi có thể giúp gì?", time: timeNow }
    ]
  };
  chatData.push(newChat);
  currentChatId = newId;
  saveToCookie();
  renderChatList();
  renderMessages();
}

function renderChatList() {
  const listEl = document.getElementById("conversationList");
  listEl.innerHTML = "";

  chatData.forEach(chat => {
    const item = document.createElement("div");
    item.classList.add("conversation-item");
    if (chat.id === currentChatId) {
      item.classList.add("active");
    }

    item.addEventListener("click", () => {
      switchChat(chat.id);
    });

    const titleSpan = document.createElement("span");
    titleSpan.textContent = chat.title;
    titleSpan.style.cursor = "pointer";
    item.appendChild(titleSpan);

    const menuContainer = document.createElement("div");
    menuContainer.style.position = "relative";
    menuContainer.style.display = "inline-block";

    const menuBtn = document.createElement("button");
    menuBtn.classList.add("btn-rename");
    menuBtn.textContent = "⋮";
    menuBtn.title = "Tùy chọn";
    menuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      menuList.classList.toggle("show");
    });
    menuContainer.appendChild(menuBtn);

    const menuList = document.createElement("div");
    menuList.classList.add("dropdown-menu");

    const renameOption = document.createElement("div");
    renameOption.classList.add("dropdown-item");
    renameOption.textContent = "Đổi tên";
    renameOption.addEventListener("click", (e) => {
      e.stopPropagation();
      menuList.classList.remove("show");
      const newTitle = prompt("Nhập tiêu đề mới:", chat.title);
      if (newTitle && newTitle.trim() !== "") {
        chat.title = newTitle.trim();
        saveToCookie();
        renderChatList();
      }
    });
    menuList.appendChild(renameOption);

    const deleteOption = document.createElement("div");
    deleteOption.classList.add("dropdown-item");
    deleteOption.textContent = "Xóa";
    deleteOption.addEventListener("click", (e) => {
      e.stopPropagation();
      menuList.classList.remove("show");
      const confirmDelete = confirm("Bạn có chắc chắn muốn xóa hội thoại này?");
      if (confirmDelete) {
        chatData = chatData.filter(c => c.id !== chat.id);
        saveToCookie();
        if (currentChatId === chat.id) {
          if (chatData.length > 0) {
            currentChatId = chatData[0].id;
            renderMessages();
          } else {
            currentChatId = null;
            document.getElementById("chatMessages").innerHTML = "";
          }
        }
        renderChatList();
      }
    });
    menuList.appendChild(deleteOption);

    menuContainer.appendChild(menuList);
    item.appendChild(menuContainer);

    listEl.appendChild(item);
  });
}

function switchChat(chatId) {
  currentChatId = chatId;
  renderChatList();
  renderMessages();
  pendingImages = [];
  renderPreviewImages();
}

function renderMessages() {
  const chatMessagesEl = document.getElementById("chatMessages");
  chatMessagesEl.innerHTML = "";

  const chatBox = chatData.find(c => c.id === currentChatId);
  if (!chatBox) return;

  chatBox.messages.forEach(msg => {
    const rowDiv = document.createElement("div");
    rowDiv.classList.add("message-row", msg.role);

    const bubbleDiv = document.createElement("div");
    bubbleDiv.classList.add("message-bubble");

    if (msg.time) {
      const timeSpan = document.createElement("small");
      timeSpan.style.display = "block";
      timeSpan.style.opacity = "0.7";
      timeSpan.textContent = msg.time;
      bubbleDiv.appendChild(timeSpan);
    }

    if (msg.images && msg.images.length > 0) {
      const imageWrapper = document.createElement("div");
      imageWrapper.style.display = "flex";
      imageWrapper.style.flexWrap = "wrap";
      imageWrapper.style.gap = "8px";
      imageWrapper.style.marginTop = "8px";

      msg.images.forEach((imageData) => {
        const img = document.createElement("img");
        img.src = imageData;
        img.classList.add("message-img");
        img.style.width = "200px";
        img.style.height = "200px";
        img.style.objectFit = "cover";
        img.style.borderRadius = "5px";
        imageWrapper.appendChild(img);
      });

      bubbleDiv.appendChild(imageWrapper);
    }

    if (msg.content) {
      const textWrapper = document.createElement("div");
      textWrapper.style.marginTop = "8px";

      const p = document.createElement("p");
      p.style.margin = "0";
      p.innerHTML = msg.content;
      textWrapper.appendChild(p);

      bubbleDiv.appendChild(textWrapper);
    }

    rowDiv.appendChild(bubbleDiv);
    chatMessagesEl.appendChild(rowDiv);
  });

  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

/**********************
 * PREVIEW ẢNH TẠM
 **********************/
function renderPreviewImages() {
  const previewContainer = document.getElementById("previewContainer");
  previewContainer.innerHTML = "";

  pendingImages.forEach((imgData, index) => {
    const wrapper = document.createElement("div");
    wrapper.classList.add("preview-item");

    const img = document.createElement("img");
    img.src = imgData;
    wrapper.appendChild(img);

    const removeBtn = document.createElement("div");
    removeBtn.classList.add("remove-btn");
    removeBtn.textContent = "x";
    removeBtn.addEventListener("click", () => {
      pendingImages.splice(index, 1);
      renderPreviewImages();
    });

    wrapper.appendChild(removeBtn);
    previewContainer.appendChild(wrapper);
  });
}

function handleImagePreview(base64Data) {
  pendingImages.push(base64Data);
  renderPreviewImages();
}

/**********************
 * GỬI TIN NHẮN
 **********************/
function sendMessage() {
  const inputEl = document.getElementById("chatInput");
  const text = inputEl.value.trim();
  if (!text && pendingImages.length === 0) return;

  const timeNow = new Date().toLocaleTimeString("vi-VN", { hour: '2-digit', minute: '2-digit' });
  const chatBox = chatData.find(c => c.id === currentChatId);
  if (!chatBox) return;

  const userMessage = {
    role: "user",
    content: text,
    images: [...pendingImages],
    time: timeNow
  };

  chatBox.messages.push(userMessage);
  inputEl.value = "";
  pendingImages = [];
  renderPreviewImages();
  renderMessages();
  saveToCookie();

  const payload = {
    user_input: text,
    images: userMessage.images
  };

  fetch('/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then(response => response.json())
    .then(data => {
      const currentChat = chatData.find(c => c.id === currentChatId);
      if (currentChat) {
        currentChat.messages = data.chat_history;
        renderMessages();
        saveToCookie();
      }
    })
    .catch(error => {
      console.error('Lỗi khi gọi pipeline:', error);
    });
}

/**********************
 * EVENT HANDLERS
 **********************/
document.getElementById("btnNewChat").addEventListener("click", createNewConversation);
document.getElementById("btnSend").addEventListener("click", sendMessage);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendMessage();
  }
});

document.getElementById("btnUpload").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});
document.getElementById("fileInput").addEventListener("change", () => {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    handleImagePreview(e.target.result);
  };
  reader.readAsDataURL(file);
  fileInput.value = "";
});

document.addEventListener("paste", (e) => {
  const clipboardItems = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (let i = 0; i < clipboardItems.length; i++) {
    const item = clipboardItems[i];
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      const reader = new FileReader();
      reader.onload = function(event) {
        handleImagePreview(event.target.result);
      };
      reader.readAsDataURL(blob);
    }
  }
});

/**********************
 * KHỞI TẠO ỨNG DỤNG
 **********************/
loadFromCookie();
renderChatList();
if (chatData.length === 0) {
  createNewConversation();
} else {
  currentChatId = chatData[0].id;
  renderMessages();
}
