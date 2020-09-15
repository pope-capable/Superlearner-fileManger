const UserServices = require('../services/userServices')
const NotificationServices = require('../services/notificationservice')


function sockets(io) {
    io.on('connection', (socket) => {
      // console.log("MEEK", socket.handshake.query)
        UserServices.updateUser({userId: socket.handshake.query.userId, active_socket: socket.id }).then(newUser => {
          // NotificationServices.findUnreadNotifications({userId: socket.handshake.query.userId}).then(foundNotes => {
          //   var notesCount = foundNotes.length
          setTimeout(() => {
            io.sockets.to(socket.id).emit('Notification', {
              title : "Welcome", content: "Onboarding completed, yay!"
          })
          }, 10000);
            // })
        })
    })
	
}

module.exports = sockets;